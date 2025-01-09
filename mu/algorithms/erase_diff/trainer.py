# mu/algorithms/erase_diff/trainer.py

import torch
import gc
from tqdm import tqdm
import random
from torch.nn import MSELoss
import wandb
import logging

from timm.utils import AverageMeter
import logging 

from mu.core import BaseTrainer
from mu.algorithms.erase_diff.model import EraseDiffModel

class EraseDiffTrainer(BaseTrainer):
    """
    Trainer for the EraseDiff algorithm.
    Handles the training loop, loss computation, and optimization.
    """

    def __init__(self, model: EraseDiffModel, config: dict, device: str,  data_handler, **kwargs):
        """
        Initialize the EraseDiffTrainer.

        Args:
            model (EraseDiffModel): Instance of EraseDiffModel.
            config (dict): Configuration dictionary.
            device (str): Device to perform training on.
            device_orig (str): Device for the original (frozen) model.
            data_handler (EraseDiffDataHandler): Instance of EraseDiffDataHandler.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model, config, **kwargs)
        self.device = device
        self.model = model.model
        self.sampler = None
        self.sampler_orig = None
        self.criteria = MSELoss()
        self.logger = logging.getLogger(__name__)
        self.data_handler = data_handler
        self.setup_optimizer()

    def setup_optimizer(self):
        """
        Setup the optimizer based on the training method.
        """
        # Select parameters to train based on train_method
        train_method = self.config.get('train_method', 'xattn')
        parameters = []
        for name, param in self.model.model.named_parameters():
            if train_method == 'full':
                parameters.append(param)
            elif train_method == 'xattn' and 'attn2' in name:
                parameters.append(param)
            elif train_method == 'selfattn' and 'attn1' in name:
                parameters.append(param)
            elif train_method == 'noxattn':
                if not (name.startswith('out.') or 'attn2' in name or 'time_embed' in name):
                    parameters.append(param)
            elif train_method == 'xlayer':
                if 'attn2' in name and ('output_blocks.6.' in name or 'output_blocks.8.' in name):
                    parameters.append(param)
            elif train_method == 'selflayer':
                if 'attn1' in name and ('input_blocks.4.' in name or 'input_blocks.7.' in name):
                    parameters.append(param)

        self.optimizer = torch.optim.Adam(parameters, lr=self.config.get('lr', 1e-5))

    def train(self):
        """
        Execute the training loop.
        """
        epochs = self.config.get('epochs', 1)
        K_steps = self.config.get('K_steps', 2)
        alpha = self.config.get('alpha', 0.1)

        # Retrieve data loaders
        data_loaders = self.data_handler.get_data_loaders()
        forget_dl = data_loaders.get('forget')
        remain_dl = data_loaders.get('remain')

        self.logger.info(f"Number of forget samples: {len(forget_dl.dataset)}")
        self.logger.info(f"Number of remain samples: {len(remain_dl.dataset)}")

        for epoch in range(epochs):
            self.logger.info(f"Starting Epoch {epoch+1}/{epochs}")
            with tqdm(total=len(forget_dl), desc=f'Epoch {epoch+1}/{epochs}') as pbar:
                self.model.train()
                param_i = self.get_param()

                for step in range(K_steps): 
                    unl_losses = AverageMeter()
                    for forget_batch in forget_dl:
                        self.optimizer.zero_grad()

                        forget_images, forget_prompts = forget_batch
                        try:
                            remain_batch = next(iter(remain_dl))
                            remain_images, remain_prompts = remain_batch
                        except StopIteration:
                            remain_dl = iter(remain_dl)
                            remain_batch = next(remain_dl)
                            remain_images, remain_prompts = remain_batch

                        pseudo_prompts = remain_prompts

                        # Forget stage
                        forget_loss = self.compute_forget_loss(forget_images, forget_prompts, pseudo_prompts)

                        forget_loss.backward()
                        self.optimizer.step()

                        unl_losses.update(forget_loss)

                    self.set_param(param_i)  # Restore original parameters

                    # Remain stage
                    for remain_batch in remain_dl:
                        self.model.train()
                        self.optimizer.zero_grad()

                        remain_images, remain_prompts = remain_batch
                        try:
                            forget_batch = next(iter(forget_dl))
                            forget_images, forget_prompts = forget_batch
                        except StopIteration:
                            forget_dl = iter(forget_dl)
                            forget_batch = next(forget_dl)
                            forget_images, forget_prompts = forget_batch

                        pseudo_prompts = remain_prompts

                        remain_btch = {
                            "edited": remain_images.to(self.device),
                            "edit": {"c_crossattn": remain_prompts}
                        }
                        # Remain loss
                        remain_loss = self.model.shared_step(remain_btch)[0]

                        # Forget loss within remain stage
                        unlearn_loss = self.compute_unlearn_loss(forget_images, forget_prompts, pseudo_prompts)

                        q_loss = unlearn_loss - unl_losses.avg.detach()

                        total_loss = remain_loss + alpha * q_loss
                        total_loss.backward()
                        self.optimizer.step()

                        # Logging
                        wandb.log({"loss": total_loss.item()})
                        pbar.set_postfix({"loss": total_loss.item() / self.config.get('batch_size', 4)})
                        pbar.update(1)

            self.logger.info(f"Epoch {epoch+1}/{epochs} completed.")

        self.model.model.eval()
        self.logger.info("Training completed.")
        return self.model.model

    def get_param(self) -> list:
        """
        Clone model parameters.

        Returns:
            list: List of cloned parameters.
        """
        new_param = []
        with torch.no_grad():
            for name, param in self.model.model.named_parameters():
                new_param.append(param.clone())
        torch.cuda.empty_cache()
        torch.manual_seed(0)
        return new_param

    def set_param(self, old_param: list):
        """
        Set model parameters from a cloned list.

        Args:
            old_param (list): List of cloned parameters.
        """
        with torch.no_grad():
            for idx, (name, param) in enumerate(self.model.model.named_parameters()):
                param.copy_(old_param[idx])
        torch.cuda.empty_cache()
        torch.manual_seed(0)

    def compute_forget_loss(self, forget_images: torch.Tensor, forget_prompts: list, pseudo_prompts: list) -> torch.Tensor:
        """
        Compute the forget loss.

        Args:
            forget_images (torch.Tensor): Batch of forget images.
            forget_prompts (list): Corresponding prompts for forget images.
            pseudo_prompts (list): Pseudo prompts derived from remain prompts.

        Returns:
            torch.Tensor: Computed forget loss.
        """
        forget_batch = {
            "edited": forget_images.to(self.device),
            "edit": {"c_crossattn": forget_prompts}
        }
        pseudo_batch = {
            "edited": forget_images.to(self.device),
            "edit": {"c_crossattn": pseudo_prompts}
        }

        forget_input, forget_emb = self.model.get_input(forget_batch, self.model.first_stage_key)
        pseudo_input, pseudo_emb = self.model.get_input(pseudo_batch, self.model.first_stage_key)

        t = torch.randint(0, self.model.num_timesteps, (forget_input.shape[0],), device=self.device).long()
        noise = torch.randn_like(forget_input, device=self.device)

        forget_noisy = self.model.q_sample(x_start=forget_input, t=t, noise=noise)
        pseudo_noisy = self.model.q_sample(x_start=pseudo_input, t=t, noise=noise)

        forget_out = self.model.apply_model(forget_noisy, t, forget_emb)
        pseudo_out = self.model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

        forget_loss = self.criteria(forget_out, pseudo_out)
        return forget_loss

    def compute_unlearn_loss(self, forget_images: torch.Tensor, forget_prompts: list, pseudo_prompts: list) -> torch.Tensor:
        """
        Compute the unlearn loss within the remain stage.

        Args:
            forget_images (torch.Tensor): Batch of forget images.
            forget_prompts (list): Corresponding prompts for forget images.
            pseudo_prompts (list): Pseudo prompts derived from remain prompts.

        Returns:
            torch.Tensor: Computed unlearn loss.
        """
        forget_batch = {
            "edited": forget_images.to(self.device),
            "edit": {"c_crossattn": forget_prompts}
        }
        pseudo_batch = {
            "edited": forget_images.to(self.device),
            "edit": {"c_crossattn": pseudo_prompts}
        }

        forget_input, forget_emb = self.model.get_input(forget_batch, self.model.first_stage_key)
        pseudo_input, pseudo_emb = self.model.get_input(pseudo_batch, self.model.first_stage_key)

        t = torch.randint(0, self.model.num_timesteps, (forget_input.shape[0],), device=self.device).long()
        noise = torch.randn_like(forget_input, device=self.device)

        forget_noisy = self.model.q_sample(x_start=forget_input, t=t, noise=noise)
        pseudo_noisy = self.model.q_sample(x_start=pseudo_input, t=t, noise=noise)

        forget_out = self.model.apply_model(forget_noisy, t, forget_emb)
        pseudo_out = self.model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

        unlearn_loss = self.criteria(forget_out, pseudo_out)
        return unlearn_loss

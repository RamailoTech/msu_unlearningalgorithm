import os
from torch.nn import MSELoss
import torch
from tqdm import tqdm
import wandb
import gc

from .model import EraseDiffModel
from .data_handler import setup_erase_diff_data
from .utils import get_param, set_param
from mu.core.base_trainer import BaseTrainer
from timm.utils import AverageMeter
from mu.datasets.utils import get_logger

class EraseDiffTrainer(BaseTrainer):
    """
    Trainer class for the EraseDiff algorithm.
    Encapsulates the training loop and related functionalities.
    """
    def __init__(
        self,
        config: dict,
        device: str,
        device_orig: str,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.device_orig = device_orig
        self.logger = get_logger(self.__class__.__name__)
        self.model = self.setup_model()
        self.forget_dl, self.remain_dl = self.setup_data_loaders()
        self.optimizer = self.setup_optimizer()
        self.criteria = MSELoss()
        self.logger.info("EraseDiffTrainer initialized.")
    
    def setup_model(self):
        """
        Initialize the model using the configuration.

        Returns:
            EraseDiffModel: Initialized model.
        """
        config_path = self.config['config_path']
        ckpt_path = self.config['ckpt_path']
        model = EraseDiffModel(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=self.device
        )
        self.logger.info("Model setup completed.")
        return model
    
    def setup_data_loaders(self):
        """
        Setup forget and remain data loaders based on configuration.

        Returns:
            tuple: Data loaders for forget and remain datasets.
        """
        forget_data_dir = os.path.join(self.config['forget_data_dir'], self.config['theme'])
        remain_data_dir = os.path.join(self.config['remain_data_dir'], 'Seed_Images')
        batch_size = self.config['batch_size']
        image_size = self.config['image_size']
        interpolation = self.config.get('interpolation', 'bicubic')
        num_workers = self.config.get('num_workers', 4)
        pin_memory = self.config.get('pin_memory', True)
        additional_param = self.config.get('additional_param', None)

        self.logger.info(f"Setting up forget data loader from {forget_data_dir}")
        forget_dl = setup_erase_diff_data(
            data_dir=forget_data_dir,
            batch_size=batch_size,
            image_size=image_size,
            interpolation=interpolation,
            num_workers=num_workers,
            pin_memory=pin_memory,
            additional_param=additional_param
        )

        self.logger.info(f"Setting up remain data loader from {remain_data_dir}")
        remain_dl = setup_erase_diff_data(
            data_dir=remain_data_dir,
            batch_size=batch_size,
            image_size=image_size,
            interpolation=interpolation,
            num_workers=num_workers,
            pin_memory=pin_memory,
            additional_param=additional_param
        )

        self.logger.info(f"Forget DataLoader: {len(forget_dl)} batches")
        self.logger.info(f"Remain DataLoader: {len(remain_dl)} batches")

        return forget_dl, remain_dl

    def setup_optimizer(self):
        """
        Setup optimizer based on the training method.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        train_method = self.config['train_method']
        parameters = []
        for name, param in self.model.model.diffusion_model.named_parameters():
            if train_method == 'noxattn':
                if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                    continue
                else:
                    parameters.append(param)
            elif train_method == 'selfattn':
                if 'attn1' in name:
                    parameters.append(param)
            elif train_method == 'xattn':
                if 'attn2' in name:
                    parameters.append(param)
            elif train_method == 'full':
                parameters.append(param)
            elif train_method == 'notime':
                if not (name.startswith('out.') or 'time_embed' in name):
                    parameters.append(param)
            elif train_method == 'xlayer':
                if 'attn2' in name and ('output_blocks.6.' in name or 'output_blocks.8.' in name):
                    parameters.append(param)
            elif train_method == 'selflayer':
                if 'attn1' in name and ('input_blocks.4.' in name or 'input_blocks.7.' in name):
                    parameters.append(param)
        
        optimizer = torch.optim.Adam(parameters, lr=self.config['lr'])
        self.logger.info(f"Optimizer initialized with train_method={train_method}")
        return optimizer

    def train(self):
        """
        Execute the training loop for style removal.
        """
        epochs = self.config['epochs']
        K_steps = self.config['K_steps']
        alpha = self.config['alpha']
        batch_size = self.config['batch_size']
        device = self.device
        image_size = self.config['image_size']
        train_method = self.config['train_method']

        for epoch in range(epochs):
            self.model.model.train()
            pbar = tqdm(range(len(self.forget_dl)), desc=f'Epoch {epoch+1}/{epochs}')

            param_i = get_param(self.model.model)

            for j in range(K_steps):
                unl_losses = AverageMeter()
                for i, _ in enumerate(self.forget_dl):
                    self.optimizer.zero_grad()

                    try:
                        forget_images, forget_prompts = next(iter(self.forget_dl))
                        remain_images, remain_prompts = next(iter(self.remain_dl))
                    except StopIteration:
                        self.forget_dl, self.remain_dl = self.setup_data_loaders()
                        forget_images, forget_prompts = next(iter(self.forget_dl))
                        remain_images, remain_prompts = next(iter(self.remain_dl))

                    # Convert prompts to lists
                    forget_prompts = list(forget_prompts)
                    remain_prompts = list(remain_prompts)
                    pseudo_prompts = remain_prompts

                    # Forget stage
                    forget_batch = {
                        "edited": forget_images.to(device),
                        "edit": {"c_crossattn": forget_prompts}
                    }

                    pseudo_batch = {
                        "edited": forget_images.to(device),
                        "edit": {"c_crossattn": pseudo_prompts}
                    }

                    forget_input, forget_emb = self.model.get_input(forget_batch, self.config['first_stage_key'])
                    pseudo_input, pseudo_emb = self.model.get_input(pseudo_batch, self.config['first_stage_key'])

                    t = torch.randint(0, self.model.model.num_timesteps, (forget_input.shape[0],), device=device).long()
                    noise = torch.randn_like(forget_input, device=device)

                    forget_noisy = self.model.q_sample(x_start=forget_input, t=t, noise=noise)
                    pseudo_noisy = self.model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                    forget_out = self.model.apply_model(forget_noisy, t, forget_emb)
                    pseudo_out = self.model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                    forget_loss = self.criteria(forget_out, pseudo_out)
                    forget_loss.backward()

                    self.optimizer.step()
                    unl_losses.update(forget_loss.item())
                    torch.cuda.empty_cache()
                    gc.collect()

                self.model = set_param(self.model, param_i)  # Reset model parameters

                # Remain stage
                for i, _ in enumerate(self.forget_dl):
                    self.model.model.train()
                    self.optimizer.zero_grad()

                    try:
                        forget_images, forget_prompts = next(iter(self.forget_dl))
                        remain_images, remain_prompts = next(iter(self.remain_dl))
                    except StopIteration:
                        self.forget_dl, self.remain_dl = self.setup_data_loaders()
                        forget_images, forget_prompts = next(iter(self.forget_dl))
                        remain_images, remain_prompts = next(iter(self.remain_dl))

                    forget_prompts = list(forget_prompts)
                    remain_prompts = list(remain_prompts)
                    pseudo_prompts = remain_prompts

                    # Remain stage
                    remain_batch = {
                        "edited": remain_images.to(device),
                        "edit": {"c_crossattn": remain_prompts}
                    }
                    remain_loss = self.model.shared_step(remain_batch)[0]

                    # Forget stage
                    forget_batch = {
                        "edited": forget_images.to(device),
                        "edit": {"c_crossattn": forget_prompts}
                    }

                    pseudo_batch = {
                        "edited": forget_images.to(device),
                        "edit": {"c_crossattn": pseudo_prompts}
                    }

                    forget_input, forget_emb = self.model.get_input(forget_batch, self.config['first_stage_key'])
                    pseudo_input, pseudo_emb = self.model.get_input(pseudo_batch, self.config['first_stage_key'])

                    t = torch.randint(0, self.model.model.num_timesteps, (forget_input.shape[0],), device=device).long()
                    noise = torch.randn_like(forget_input, device=device)

                    forget_noisy = self.model.q_sample(x_start=forget_input, t=t, noise=noise)
                    pseudo_noisy = self.model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                    forget_out = self.model.apply_model(forget_noisy, t, forget_emb)
                    pseudo_out = self.model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                    unlearn_loss = self.criteria(forget_out, pseudo_out)
                    q_loss = unlearn_loss - torch.tensor(unl_losses.avg).detach()

                    total_loss = remain_loss + alpha * q_loss
                    total_loss.backward()
                    self.optimizer.step()

                    wandb.log({"total_loss": total_loss.item()})
                    pbar.set_postfix({"loss": total_loss.item() / batch_size})

            self.logger.info(f"Epoch {epoch+1}/{epochs} completed.")

        self.model.model.eval()
        self.logger.info("Training completed.")

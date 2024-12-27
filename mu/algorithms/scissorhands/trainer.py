import torch
from tqdm import tqdm
from torch.nn import MSELoss
import logging
import copy

from mu.core import BaseTrainer
from mu.algorithms.scissorhands.model import ScissorHandsModel
from mu.algorithms.scissorhands.data_handler import ScissorHandsDataHandler

from mu.algorithms.scissorhands.utils import snip, project2cone2

class ScissorHandsTrainer(BaseTrainer):
    """
    Trainer for the ScissorHands algorithm.
    Handles the training loop, loss computation, and optimization.
    """

    def __init__(self, model: ScissorHandsModel, config: dict, device: str,  data_handler: ScissorHandsDataHandler, **kwargs):
        """
        Initialize the ScissorHandsTrainer.

        Args:
            model: Instance of the ScissorHands model.
            config: Configuration dictionary.
            data_handler: Instance of ScissorHandsDataHandler for data handling.
            device: Device to use for training (e.g., 'cuda:0').
        """
        super().__init__(model, config, **kwargs)
        self.device = device
        self.model = model.model
        self.criteria = MSELoss()
        self.logger = logging.getLogger(__name__)
        self.data_handler = data_handler
        self.setup_optimizer()


    def setup_optimizer(self):
        """
        Setup the optimizer for the training process.
        """
        parameters = self.select_parameters()
        self.optimizer = torch.optim.Adam(parameters, lr=float(self.config.get('lr', 1e-5)))

    def select_parameters(self):
        """
        Select parameters to train based on the specified training method.
        """
        train_method = self.config.get('train_method', 'xattn')
        parameters = []

        for name, param in self.model.named_parameters():
            if train_method == 'full':
                parameters.append(param)
            elif train_method == 'xattn' and 'attn2' in name:
                parameters.append(param)
            elif train_method == 'selfattn' and 'attn1' in name:
                parameters.append(param)
            elif train_method == 'noxattn':
                if not (name.startswith('out.') or 'attn2' in name or 'time_embed' in name):
                    parameters.append(param)
            elif train_method == 'xlayer' and 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    parameters.append(param)
            elif train_method == 'selflayer' and 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    parameters.append(param)
        return parameters
    

    def train(self):
        """
        Execute the training loop.
        """
        epochs = self.config.get('epochs', 2)
        sparsity = self.config.get('sparsity', 0.99)
        prune_num = self.config.get('prune_num', 1)
        project = self.config.get('project', False)
        alpha = self.config.get('alpha', 0.1)

        # Retrieve data loaders
        data_loaders = self.data_handler.get_data_loaders()
        forget_dl = data_loaders.get('forget')
        remain_dl = data_loaders.get('remain')
        self.logger.info(f"Forget dataset size: {len(forget_dl)}")
        self.logger.info(f"Remain dataset size: {len(remain_dl)}")

        # Prune using SNIP
        self.logger.info("Applying SNIP pruning...")
        self.model = snip(self.model, forget_dl, sparsity, prune_num, self.device)

        # Projection setup if required
        g_o = None
        if project:
            g_o = self.prepare_projection(forget_dl)

        self.logger.info("Starting training...")
        for epoch in range(epochs):
            with tqdm(total=len(forget_dl), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for forget_batch, remain_batch in zip(forget_dl, remain_dl):
                    self.optimizer.zero_grad()

                    forget_images, forget_prompts = forget_batch
                    remain_images, remain_prompts = remain_batch

                    forget_loss = self.compute_forget_loss(forget_images, forget_prompts, remain_prompts)
                    remain_loss = self.compute_remain_loss(remain_images, remain_prompts)

                    total_loss = alpha * forget_loss + remain_loss
                    total_loss.backward()

                    if project:
                        self.apply_projection(g_o)

                    self.optimizer.step()
                    pbar.set_postfix({"loss": total_loss.item()})
                    pbar.update(1)

        self.logger.info("Training completed.")
        return self.model

    def compute_forget_loss(self, forget_images, forget_prompts, pseudo_prompts):
        """
        Compute the forget loss.

        Args:
            forget_images: Batch of forget images.
            forget_prompts: Prompts corresponding to forget images.
            pseudo_prompts: Pseudo-prompts generated from remain prompts.

        Returns:
            forget_loss: Loss for forgetting.
        """
        forget_batch = {"edited": forget_images.to(self.device), "edit": {"c_crossattn": forget_prompts}}
        pseudo_batch = {"edited": forget_images.to(self.device), "edit": {"c_crossattn": pseudo_prompts}}

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

    def compute_remain_loss(self, remain_images, remain_prompts):
        """
        Compute the remain loss.

        Args:
            remain_images: Batch of remain images.
            remain_prompts: Prompts corresponding to remain images.

        Returns:
            remain_loss: Loss for remaining images.
        """
        remain_batch = {"edited": remain_images.to(self.device), "edit": {"c_crossattn": remain_prompts}}
        remain_loss = self.model.shared_step(remain_batch)[0]
        return remain_loss

    def prepare_projection(self, forget_dl):
        """
        Prepare projection gradients using forget data loader.

        Args:
            forget_dl: Forget data loader.

        Returns:
            g_o: Gradients for projection.
        """
        proxy_model = copy.deepcopy(self.model).to(self.device)
        proxy_model.eval()
        g_o = []

        for forget_images, forget_prompts in forget_dl:
            forget_batch = {"edited": forget_images.to(self.device), "edit": {"c_crossattn": forget_prompts}}
            loss = -proxy_model.shared_step(forget_batch)[0]
            loss.backward()

            grad_o = [param.grad.detach().view(-1) for name, param in proxy_model.model.named_parameters()
                      if param.grad is not None and 'attn2' in name]
            g_o.append(torch.cat(grad_o))

        g_o = torch.stack(g_o, dim=1)
        return g_o

    def apply_projection(self, g_o):
        """
        Apply gradient projection if required.

        Args:
            g_o: Gradients for projection.
        """
        grad_f = [param.grad for name, param in self.model.model.named_parameters()
                  if param.grad is not None and 'attn2' in name]
        g_f = torch.cat([g.view(-1) for g in grad_f])

        dotg = torch.mm(g_f.unsqueeze(0), g_o)
        if (dotg < 0).sum() != 0:
            grad_new = project2cone2(g_f.unsqueeze(0), g_o)
            pointer = 0
            for param in grad_f:
                grad_shape = param.grad.shape
                this_grad = grad_new[pointer:pointer + param.numel()].view(grad_shape).to(self.device)
                param.grad.data.copy_(this_grad)
                pointer += param.numel()

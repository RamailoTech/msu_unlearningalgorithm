# erasediff_trainer.py

from algorithms.core.base_trainer import BaseTrainer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from timm.utils import AverageMeter
import gc

class EraseDiffTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, config, data_handler, wandb=None):
        super().__init__(model, optimizer, criterion, config)
        self.data_handler = data_handler
        self.wandb = wandb
        self.device = config.get('device', 'cuda:0')
        self.train_method = config['train_method']
        self.alpha = config.get('alpha', 0.1)
        self.K_steps = config.get('K_steps', 2)
        self.batch_size = config.get('batch_size', 4)
        self.epochs = config.get('epochs', 5)
        self.image_size = config.get('image_size', 512)
        self.model = self.model.model  # Access the inner model
        self.configure_parameters()

    def configure_parameters(self):
        parameters = []
        for name, param in self.model.diffusion_model.named_parameters():
            # Implement parameter selection logic as per train_method
            if self.train_method == 'noxattn':
                if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                    pass
                else:
                    parameters.append(param)
            elif self.train_method == 'selfattn':
                if 'attn1' in name:
                    parameters.append(param)
            elif self.train_method == 'xattn':
                if 'attn2' in name:
                    parameters.append(param)
            elif self.train_method == 'full':
                parameters.append(param)
            # Add other methods as needed
        self.optimizer = torch.optim.Adam(parameters, lr=self.config['lr'])

    def train(self):
        forget_dl, remain_dl = self.data_handler.get_data_loaders()
        self.model.train()

        for epoch in range(self.epochs):
            self.model.train()
            with tqdm(total=len(forget_dl), desc=f'Epoch {epoch+1}/{self.epochs}') as progress_bar:
                for j in range(self.K_steps):
                    unl_losses = AverageMeter()
                    # **First Stage: Forgetting**
                    param_i = self.get_model_params()  # Save initial parameters

                    for batch_idx, (forget_data, _) in enumerate(forget_dl):
                        self.optimizer.zero_grad()
                        # Get batch data
                        forget_images, forget_prompts = forget_data
                        remain_images, remain_prompts = next(iter(remain_dl))

                        # Prepare batches
                        forget_batch = {
                            "edited": forget_images.to(self.device),
                            "edit": {"c_crossattn": list(forget_prompts)}
                        }
                        pseudo_batch = {
                            "edited": forget_images.to(self.device),
                            "edit": {"c_crossattn": list(remain_prompts)}
                        }

                        # Get inputs and embeddings
                        forget_input, forget_emb = self.model.get_input(forget_batch, self.model.first_stage_key)
                        pseudo_input, pseudo_emb = self.model.get_input(pseudo_batch, self.model.first_stage_key)

                        # Noise and time steps
                        t = torch.randint(0, self.model.num_timesteps, (forget_input.shape[0],), device=self.device).long()
                        noise = torch.randn_like(forget_input, device=self.device)

                        # Forward pass
                        forget_noisy = self.model.q_sample(x_start=forget_input, t=t, noise=noise)
                        pseudo_noisy = self.model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                        forget_out = self.model.apply_model(forget_noisy, t, forget_emb)
                        pseudo_out = self.model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                        # Compute loss
                        forget_loss = self.criterion(forget_out, pseudo_out)
                        forget_loss.backward()
                        self.optimizer.step()
                        unl_losses.update(forget_loss.item())

                        torch.cuda.empty_cache()
                        gc.collect()

                    # Restore parameters
                    self.set_model_params(param_i)

                    # **Second Stage: Remain**
                    for batch_idx, (forget_data, _) in enumerate(forget_dl):
                        self.optimizer.zero_grad()
                        # Get batch data
                        forget_images, forget_prompts = forget_data
                        remain_images, remain_prompts = next(iter(remain_dl))

                        # Prepare batches
                        remain_batch = {
                            "edited": remain_images.to(self.device),
                            "edit": {"c_crossattn": list(remain_prompts)}
                        }
                        remain_loss = self.model.shared_step(remain_batch)[0]

                        # Forget batches
                        forget_batch = {
                            "edited": forget_images.to(self.device),
                            "edit": {"c_crossattn": list(forget_prompts)}
                        }
                        pseudo_batch = {
                            "edited": forget_images.to(self.device),
                            "edit": {"c_crossattn": list(remain_prompts)}
                        }

                        # Get inputs and embeddings
                        forget_input, forget_emb = self.model.get_input(forget_batch, self.model.first_stage_key)
                        pseudo_input, pseudo_emb = self.model.get_input(pseudo_batch, self.model.first_stage_key)

                        # Noise and time steps
                        t = torch.randint(0, self.model.num_timesteps, (forget_input.shape[0],), device=self.device).long()
                        noise = torch.randn_like(forget_input, device=self.device)

                        # Forward pass
                        forget_noisy = self.model.q_sample(x_start=forget_input, t=t, noise=noise)
                        pseudo_noisy = self.model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                        forget_out = self.model.apply_model(forget_noisy, t, forget_emb)
                        pseudo_out = self.model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                        # Compute loss
                        unlearn_loss = self.criterion(forget_out, pseudo_out)
                        q_loss = unlearn_loss - unl_losses.avg

                        total_loss = remain_loss + self.alpha * q_loss
                        total_loss.backward()
                        self.optimizer.step()

                        if self.wandb:
                            self.wandb.log({'total_loss': total_loss.item()})

                        torch.cuda.empty_cache()
                        gc.collect()

                    progress_bar.update(1)

    def compute_loss(self, output: Any, target: Any) -> Any:
        # Implement if needed
        pass

    def step_optimizer(self):
        self.optimizer.step()

    def validate(self, *args, **kwargs):
        # Implement validation if needed
        pass

    def save_checkpoint(self, *args, **kwargs):
        # Implement checkpoint saving if needed
        pass

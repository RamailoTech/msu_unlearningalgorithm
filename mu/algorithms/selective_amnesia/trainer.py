import torch
from torch.nn import MSELoss
from torch.optim import Adam
from core.base_trainer import BaseTrainer
import wandb
import logging
from tqdm import tqdm
from typing import Dict

logger = logging.getLogger(__name__)

class SelectiveAmnesiaTrainer(BaseTrainer):
    """
    Trainer for the Selective Amnesia algorithm.
    Incorporates EWC loss and other SA-specific training logic.
    """

    def __init__(self, model, config: Dict, device: str, data_handler, **kwargs):
        super().__init__(model, config, **kwargs)
        self.device = device
        self.model = model
        self.data_handler = data_handler
        self.config = config
        self.criteria = MSELoss()
        self.setup_optimizer()

    def setup_optimizer(self):
        train_method = self.config.get('train_method', 'full')
        lr = self.config.get('lr', 5e-5)
        parameters = []

        for name, param in self.model.model.named_parameters():
            # Example selective finetuning logic
            if train_method == 'full':
                parameters.append(param)
            elif train_method == 'xattn' and 'attn2' in name:
                parameters.append(param)
            # Add other methods if needed

        self.optimizer = Adam(parameters, lr=lr)

    def train(self):
        epochs = self.config.get('epochs', 1)
        data_loaders = self.data_handler.get_data_loaders()
        train_dl = data_loaders.get('train')

        wandb.init(project=self.config.get('project_name', 'selective_amnesia'), 
                   name=self.config.get('run_name', 'selective_amnesia_run'),
                   config=self.config)
        logger.info("WandB logging initialized.")

        global_step = 0
        for epoch in range(epochs):
            logger.info(f"Starting Epoch {epoch+1}/{epochs}")
            self.model.model.train()
            with tqdm(total=len(train_dl), desc=f'Epoch {epoch+1}/{epochs}') as pbar:
                for batch in train_dl:
                    self.optimizer.zero_grad()

                    images = batch.to(self.device)
                    loss = self.compute_loss(images)

                    loss.backward()
                    self.optimizer.step()

                    wandb.log({"loss": loss.item(), "epoch": epoch+1, "step": global_step})
                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update(1)
                    global_step += 1

            logger.info(f"Epoch {epoch+1}/{epochs} completed.")

        # Save final model
        output_name = self.config.get('output_name', 'selective_amnesia_model.pth')
        self.model.save_model(output_name)
        logger.info(f"Trained model saved at {output_name}")
        wandb.save(output_name)
        wandb.finish()
        return self.model.model

    def compute_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss using EWC (Elastic Weight Consolidation) logic.
        This is just a placeholder. Implement your forgetting objective:
        - Possibly compute a diffusion step and measure difference
        - Add regularization terms using self.model.fim_dict
        """
        # Dummy logic:
        # You would typically do diffusion forward passes and compute EWC penalty:
        # EWC penalty: sum over parameters of FIM * (param - param_star)^2
        # Here we only return a dummy MSE loss with zero target.

        target = torch.zeros_like(images)
        mse_loss = self.criteria(images, target)

        # Add EWC penalty using self.model.fim_dict if available
        # Example (pseudo-code):
        # ewc_loss = 0
        # for name, param in self.model.model.named_parameters():
        #     if name in self.model.fim_dict:
        #         param_star = ... # original parameter before finetuning
        #         fim_val = self.model.fim_dict[name]
        #         ewc_loss += torch.sum(fim_val * (param - param_star)**2)
        # total_loss = mse_loss + lambda * ewc_loss
        # return total_loss

        return mse_loss

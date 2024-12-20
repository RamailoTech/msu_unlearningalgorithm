import torch
import wandb
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
import logging

from core.base_trainer import BaseTrainer
from typing import Dict


class ConceptAblationTrainer(BaseTrainer):
    """
    Trainer class for the Concept Ablation algorithm.
    Handles the training loop, loss computation, and optimization.
    """

    def __init__(self, model, config: Dict, device: str, data_handler, **kwargs):
        """
        Initialize the ConceptAblationTrainer.

        Args:
            model: Instance of ConceptAblationModel or a similar model class.
            config (dict): Configuration dictionary with training parameters.
            device (str): Device to perform training on (e.g. 'cuda:0').
            data_handler: An instance of ConceptAblationDataHandler or a similar data handler.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model, config, **kwargs)
        self.device = device
        self.model = model
        self.data_handler = data_handler
        self.logger = logging.getLogger(__name__)
        self.criteria = MSELoss()
        self.setup_optimizer()

    def setup_optimizer(self):
        """
        Setup the optimizer based on the training configuration.
        Adjust parameter groups or other attributes as per concept ablation needs.
        """
        train_method = self.config.get('train_method', 'full')
        parameters = []
        for name, param in self.model.model.named_parameters():
            # Example: fine-tuning cross-attn layers only
            # Adjust logic as needed for concept ablation specifics
            if train_method == 'full':
                parameters.append(param)
            elif train_method == 'xattn' and 'attn2' in name:
                parameters.append(param)
            # Add other conditions if needed
        lr = self.config.get('lr', 1e-5)
        self.optimizer = Adam(parameters, lr=lr)

    def train(self):
        """
        Execute the training loop.
        """
        epochs = self.config.get('epochs', 1)

        # Get the train dataloader
        data_loaders = self.data_handler.get_data_loaders()
        train_dl = data_loaders.get('train')

        # Initialize WandB logging if needed
        project_name = self.config.get('project_name', 'concept_ablation_project')
        run_name = self.config.get('run_name', 'concept_ablation_run')
        wandb.init(project=project_name, name=run_name, config=self.config)
        self.logger.info("WandB logging initialized.")

        self.logger.info("Starting training...")
        global_step = 0
        for epoch in range(epochs):
            self.logger.info(f"Starting Epoch {epoch+1}/{epochs}")
            self.model.model.train()
            with tqdm(total=len(train_dl), desc=f'Epoch {epoch+1}/{epochs}') as pbar:
                for batch in train_dl:
                    self.optimizer.zero_grad()

                    images, prompts = batch
                    images = images.to(self.device)

                    # Compute loss
                    loss = self.compute_loss(images, prompts)

                    loss.backward()
                    self.optimizer.step()

                    # Logging
                    wandb.log({"loss": loss.item(), "epoch": epoch+1, "step": global_step})
                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update(1)
                    global_step += 1

            self.logger.info(f"Epoch {epoch+1}/{epochs} completed.")

        self.logger.info("Training completed.")
        # Save the trained model
        output_name = self.config.get('output_name', 'concept_ablation_model.pth')
        self.model.save_model(output_name)
        self.logger.info(f"Trained model saved at {output_name}")
        wandb.save(output_name)
        wandb.finish()

        return self.model.model

    def compute_loss(self, images: torch.Tensor, prompts: list) -> torch.Tensor:
        """
        Compute the training loss for concept ablation.

        Args:
            images (torch.Tensor): Batch of images (e.g., generated or from dataset).
            prompts (list): Corresponding textual prompts.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Below is placeholder logic:
        # In practice, you'd implement the concept ablation loss, which may involve:
        # - Getting conditioning from prompts
        # - Diffusion step forward
        # - Comparing model outputs with a target distribution or image
        # For demonstration, we will:
        # 1. Get conditioning from prompts
        # 2. Apply model to a noisy latent and compute MSE with another latent (dummy logic)

        # Generate a dummy latent as a stand-in. In reality, you'd follow your pipeline's logic.
        # For instance, you might sample latents from a prior distribution or use q_sample steps.
        b, c, h, w = images.shape
        noisy_latent = torch.randn((b, 4, h // 8, w // 8), device=self.device)
        t = torch.randint(0, self.model.model.num_timesteps, (b,), device=self.device).long()
        c = self.model.get_learned_conditioning(prompts)

        # Apply model forward pass on noisy_latent
        output = self.model.apply_model(noisy_latent, t, c)

        # Dummy target (for demonstration): just use the noisy latent shifted by some factor
        target = noisy_latent * 0.0  # in practice, you'd define a meaningful target

        loss = self.criteria(output, target)
        return loss

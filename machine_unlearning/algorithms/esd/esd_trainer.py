from algorithms.core.base_trainer import BaseTrainer
import torch
from tqdm import tqdm
import random

class ESDTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, config, sampler, model_orig, sampler_orig, wandb=None):
        super().__init__(model, optimizer, criterion, config)
        self.sampler = sampler
        self.model_orig = model_orig
        self.sampler_orig = sampler_orig
        self.wandb = wandb
        self.devices = config.get('devices', ['cuda:0', 'cuda:1'])
        self.ddim_steps = config.get('ddim_steps', 50)
        self.image_size = config.get('image_size', 512)
        self.ddim_eta = 0

    def train(self, num_iterations, words, train_method, start_guidance, negative_guidance, seperator=None):
        self.model.model.train()
        parameters = self.select_parameters(train_method)
        self.optimizer = torch.optim.Adam(parameters, lr=self.config['lr'])
        criterion = self.criterion

        pbar = tqdm(range(num_iterations))
        for _ in pbar:
            word = random.choice(words)
            # Prepare embeddings
            emb_0 = self.model.get_learned_conditioning([''])
            emb_p = self.model.get_learned_conditioning([word])
            emb_n = self.model.get_learned_conditioning([f'{word}'])

            self.optimizer.zero_grad()

            # Random timestep
            t_enc = torch.randint(self.ddim_steps, (1,), device=self.devices[0])
            og_num = round((int(t_enc) / self.ddim_steps) * 1000)
            og_num_lim = round((int(t_enc + 1) / self.ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=self.devices[0])

            start_code = torch.randn((1, 4, 64, 64)).to(self.devices[0])

            with torch.no_grad():
                # Generate image with concept from original model
                z = self.quick_sample_till_t(emb_p.to(self.devices[0]), start_guidance, start_code, int(t_enc))
                # Get scores from original model
                e_0 = self.model_orig.apply_model(z.to(self.devices[1]), t_enc_ddpm.to(self.devices[1]), emb_0.to(self.devices[1]))
                e_p = self.model_orig.apply_model(z.to(self.devices[1]), t_enc_ddpm.to(self.devices[1]), emb_p.to(self.devices[1]))

            # Get scores from finetuned model
            e_n = self.model.apply_model(z.to(self.devices[0]), t_enc_ddpm.to(self.devices[0]), emb_n.to(self.devices[0]))
            e_0.requires_grad = False
            e_p.requires_grad = False

            # Compute loss
            loss = criterion(e_n.to(self.devices[0]), e_0.to(self.devices[0]) - (negative_guidance * (e_p.to(self.devices[0]) - e_0.to(self.devices[0]))))

            # Backpropagation
            loss.backward()
            if self.wandb is not None:
                self.wandb.log({"loss": loss.item()})
            pbar.set_postfix({"loss": loss.item()})
            self.optimizer.step()

    def quick_sample_till_t(self, conditioning, scale, start_code, t_enc):
        uc = None
        if scale != 1.0:
            uc = self.model.get_learned_conditioning([''])
        shape = [4, self.image_size // 8, self.image_size // 8]
        samples = self.sampler.sampler.sample(
            S=self.ddim_steps,
            conditioning=conditioning,
            batch_size=1,
            shape=shape,
            verbose=False,
            x_T=start_code,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=self.ddim_eta,
            till_T=t_enc
        )
        return samples[0]

    def select_parameters(self, train_method):
        parameters = []
        for name, param in self.model.model.model.diffusion_model.named_parameters():
            # Implement parameter selection logic as per train_method
            # ...
            pass  # Same logic as before
        return parameters

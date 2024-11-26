from algorithms.core.base_trainer import BaseTrainer
from algorithms.esd.model import ESDModel
import torch
from tqdm import tqdm
import random
from utils import load_model_from_config, sample_model
from torch.nn import MSELoss
import wandb

class ESDTrainer(BaseTrainer):
    """Trainer for the ESD algorithm."""

    def __init__(self, model: ESDModel, config: dict, device, device_orig, **kwargs):
        super().__init__(model, config, **kwargs)
        self.device = device
        self.device_orig = device_orig
        self.model_orig = None
        self.sampler = None
        self.sampler_orig = None
        self.criteria = MSELoss()
        self._setup_models()
        self._setup_optimizer()

    def _setup_models(self):
        # Load the original (frozen) model
        config_path = self.config['config_path']
        ckpt_path = self.config['ckpt_path']
        self.model_orig = load_model_from_config(config_path, ckpt_path, device=self.device_orig)
        self.model_orig.eval()

        # Setup samplers
        self.sampler = DDIMSampler(self.model.model)
        self.sampler_orig = DDIMSampler(self.model_orig)

    def _setup_optimizer(self):
        # Select parameters to train based on train_method
        train_method = self.config['train_method']
        parameters = []
        for name, param in self.model.model.model.diffusion_model.named_parameters():
            if train_method == 'full':
                parameters.append(param)
            elif train_method == 'xattn' and 'attn2' in name:
                parameters.append(param)
            elif train_method == 'selfattn' and 'attn1' in name:
                parameters.append(param)
            elif train_method == 'noxattn' and not ('attn2' in name or 'time_embed' in name):
                parameters.append(param)
            # Add other training methods as needed
        self.optimizer = torch.optim.Adam(parameters, lr=self.config['lr'])

    def train(self):
        iterations = self.config['iterations']
        ddim_steps = self.config['ddim_steps']
        start_guidance = self.config['start_guidance']
        negative_guidance = self.config['negative_guidance']
        prompt = self.config['prompt']
        seperator = self.config.get('seperator')

        # Handle multiple words if separator is provided
        if seperator:
            words = [w.strip() for w in prompt.split(seperator)]
        else:
            words = [prompt]

        self.model.model.train()
        pbar = tqdm(range(iterations))
        for i in pbar:
            word = random.choice(words)
            # Get text embeddings
            emb_0 = self.model.model.get_learned_conditioning([''])
            emb_p = self.model.model.get_learned_conditioning([word])
            emb_n = self.model.model.get_learned_conditioning([f'{word}'])

            self.optimizer.zero_grad()
            t_enc = torch.randint(ddim_steps, (1,), device=self.device)
            og_num = round((int(t_enc.item()) / ddim_steps) * 1000)
            og_num_lim = round(((int(t_enc.item()) + 1) / ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=self.device)
            start_code = torch.randn((1, 4, 64, 64)).to(self.device)

            with torch.no_grad():
                # Generate an image with the concept from the ESD model
                z = sample_model(self.model.model, self.sampler,
                                 emb_p.to(self.device), 512, 512, ddim_steps, start_guidance, 0,
                                 start_code=start_code, till_T=int(t_enc.item()), verbose=False)
                # Get conditional and unconditional scores from the frozen model
                e_0 = self.model_orig.apply_model(z.to(self.device_orig), t_enc_ddpm.to(self.device_orig), emb_0.to(self.device_orig))
                e_p = self.model_orig.apply_model(z.to(self.device_orig), t_enc_ddpm.to(self.device_orig), emb_p.to(self.device_orig))

            # Get conditional score from the ESD model
            e_n = self.model.model.apply_model(z.to(self.device), t_enc_ddpm.to(self.device), emb_n.to(self.device))
            e_0 = e_0.detach()
            e_p = e_p.detach()
            # Compute loss
            loss = self.criteria(e_n, e_0 - (negative_guidance * (e_p - e_0)))
            loss.backward()
            self.optimizer.step()

            # Logging
            wandb.log({"loss": loss.item()})
            pbar.set_postfix({"loss": loss.item()})

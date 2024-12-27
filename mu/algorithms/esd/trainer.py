import torch
from tqdm import tqdm
import random
from torch.nn import MSELoss


from mu.helpers import load_model_from_config, sample_model
from mu.core import BaseTrainer
from mu.algorithms.esd.model import ESDModel
from mu.algorithms.esd.sampler import ESDSampler

class ESDTrainer(BaseTrainer):
    """Trainer for the ESD algorithm."""

    def __init__(self, model: ESDModel, sampler: ESDSampler, config: dict, device, device_orig, **kwargs):
        super().__init__(model, config, **kwargs)
        self.device = device
        self.device_orig = device_orig
        self.model, self.model_orig = model.models
        self.sampler, self.sampler_orig = sampler.samplers
        self.criteria = MSELoss()
        self.setup_optimizer()


    def setup_optimizer(self):
        """
        Sets up the optimizer for training based on the specified training method.
        """

        train_method = self.config['train_method']
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
            elif train_method == 'notime':
                if not (name.startswith('out.') or 'time_embed' in name):
                    parameters.append(param)
            elif train_method == 'xlayer':
                if 'attn2' in name and ('output_blocks.6.' in name or 'output_blocks.8.' in name):
                    parameters.append(param)
            elif train_method == 'selflayer':
                if 'attn1' in name and ('input_blocks.4.' in name or 'input_blocks.7.' in name):
                    parameters.append(param)
        self.optimizer = torch.optim.Adam(parameters, lr=float(self.config['lr']))

    def train(self):
        """
        Execute the training loop.
        """
        iterations = self.config['iterations']
        ddim_steps = self.config['ddim_steps']
        start_guidance = self.config['start_guidance']
        negative_guidance = self.config['negative_guidance']
        prompt = self.config['prompt']
        seperator = self.config.get('seperator')

        # Prompt cleaning
        word_print = prompt.replace(' ','')
        if prompt == 'allartist':
            prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
        elif prompt == 'i2p':
            prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
        elif prompt == "artifact":
            prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

        # Handle multiple words if separator is provided
        if seperator:
            words = [w.strip() for w in prompt.split(seperator)]
        else:
            words = [prompt]

        image_size = self.config['image_size']

        self.model.train()
        pbar = tqdm(range(iterations))
        for i in pbar:
            word = random.choice(words)
            # Get text embeddings
            emb_0 = self.model.get_learned_conditioning([''])
            emb_p = self.model.get_learned_conditioning([word])
            emb_n = self.model.get_learned_conditioning([f'{word}'])

            self.optimizer.zero_grad()
            t_enc = torch.randint(ddim_steps, (1,), device=self.device)
            og_num = round((int(t_enc.item()) / ddim_steps) * 1000)
            og_num_lim = round(((int(t_enc.item()) + 1) / ddim_steps) * 1000)
            t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=self.device)
            start_code = torch.randn((1, 4, image_size // 8, image_size // 8)).to(self.device)

            with torch.no_grad():
                # Generate an image with the concept from the ESD model
                z = sample_model(self.model, self.sampler,
                                 emb_p.to(self.device), image_size, image_size, ddim_steps, start_guidance, 0,
                                 start_code=start_code, till_T=int(t_enc.item()), verbose=False)
                # Get conditional and unconditional scores from the frozen model
                e_0 = self.model_orig.apply_model(z.to(self.device_orig), t_enc_ddpm.to(self.device_orig), emb_0.to(self.device_orig))
                e_p = self.model_orig.apply_model(z.to(self.device_orig), t_enc_ddpm.to(self.device_orig), emb_p.to(self.device_orig))

            # Get conditional score from the ESD model
            e_n = self.model.apply_model(z.to(self.device), t_enc_ddpm.to(self.device), emb_n.to(self.device))
            e_0 = e_0.detach()
            e_p = e_p.detach()
            # Compute loss
            loss = self.criteria(e_n, e_0 - (negative_guidance * (e_p - e_0)))
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
        return self.model

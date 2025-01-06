import os
import torch
import numpy as np
import logging
from PIL import Image
from torch import autocast
from torch import nn
from pytorch_lightning import seed_everything
from argparse import ArgumentParser
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.constants.const import theme_available, class_available
from mu.core.base_image_generator import BaseImageGenerator
from mu.helpers.utils import load_model_from_config, calculate_fid,load_style_ref_images,load_style_generated_images
from mu.helpers import load_config
import logging
from mu.core.base_evaluator import BaseEvaluator
from torchvision import transforms
import timm
from tqdm import tqdm


class ESDImageGenerator(BaseImageGenerator):
    """ESD Image Generator class extending BaseImageGenerator."""

    def __init__(self, config: str, **kwargs):
        """Initialize the ESDImageGenerator with a YAML config."""
        # Load config and allow overrides from kwargs
        self.config = config
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sampler = None
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the model using config and initialize the sampler."""
        self.logger.info("Loading model...")
        self.model = load_model_from_config(self.config["model_config"], self.config["ckpt_path"])
        self.model = self.model.to(self.device)
        self.sampler = DDIMSampler(self.model)
        self.logger.info("Model loaded successfully.")

    def sample_image(self):
        """Sample and generate images using the ESD model based on the config."""
        steps = self.config['ddim_steps']
        theme = self.config['theme']
        cfg_text = self.config['cfg_text']
        seed = self.config['seed']
        H = self.config['image_height']
        W = self.config['image_width']
        ddim_eta = self.config['ddim_eta']
        output_dir = self.config['output_dir']

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Generating images and saving to {output_dir}")
        seed_everything(seed)

        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                with torch.no_grad():
                    with autocast(self.device):
                        with self.model.ema_scope():
                            uc = self.model.get_learned_conditioning([""])
                            c = self.model.get_learned_conditioning(prompt)
                            shape = [4, H // 8, W // 8]
                            samples_ddim, _ = self.sampler.sample(
                                S=steps, conditioning=c, batch_size=1, shape=shape,
                                verbose=False, unconditional_guidance_scale=cfg_text,
                                unconditional_conditioning=uc, eta=ddim_eta, x_T=None
                            )

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1)
                            x_sample = (255. * x_samples_ddim[0].numpy()).round().astype(np.uint8)
                            img = Image.fromarray(x_sample)
                            self.save_image(img, os.path.join(output_dir, f"{test_theme}_{object_class}_seed{seed}.jpg"))

        self.logger.info("Image generation completed.")

    def save_image(self, image, file_path):
        """Save an image to the specified path."""
        image.save(file_path)
        self.logger.info(f"Image saved at: {file_path}")

    
class ESDEvaluator(BaseEvaluator):
    """Evaluator combining Accuracy and FID metrics using ImageGenerator output."""

    def __init__(self, config):
        self.config = config
        self.generator = ESDImageGenerator(config)
        self.logger = logging.getLogger(__name__)
        self.results = {}

    def load_model(self):
        """Load the model."""
        device = self.config['device']
        model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
        num_classes = len(theme_available)
        model.head = torch.nn.Linear(1024, num_classes).to(device)
        model.load_state_dict(torch.load(self.config['ckpt_path'], map_location=device)["model_state_dict"])
        model.eval()

    def calculate_accuracy(self):
        """Calculate unlearning and retaining accuracy using the original accuracy.py logic."""
        device = self.config['device']
        theme = self.config['theme']
        input_dir = self.config['output_dir'] #output from image generation
        output_dir = self.config['accuracy_output_dir']
        output_path = os.path.join(output_dir, f"{theme}.pth") if theme is not None else os.path.join(output_dir, "result.pth")
        task = self.config['task']
        seed = self.config['seed']

        os.makedirs(output_dir, exist_ok=True)
        model = self.load_model()

        results = {}
        results["test_theme"] = theme if theme is not None else "sd"
        results["input_dir"] = self.config['output_dir']
        if task == "style":
            results["loss"] = {theme: 0.0 for theme in theme_available}
            results["acc"] = {theme: 0.0 for theme in theme_available}
            results["pred_loss"] = {theme: 0.0 for theme in theme_available}
            results["misclassified"] = {theme: {other_theme: 0 for other_theme in theme_available} for theme in theme_available}
        else:
            results["loss"] = {class_: 0.0 for class_ in class_available}
            results["acc"] = {class_: 0.0 for class_ in class_available}
            results["pred_loss"] = {class_: 0.0 for class_ in class_available}
            results["misclassified"] = {class_: {other_class: 0 for other_class in class_available} for class_ in class_available}


        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        if self.config['task'] == "style":
            for idx, test_theme in tqdm(enumerate(theme_available)):
                theme_label = idx
                for seed in seed:
                    for object_class in class_available:
                        img_path = os.path.join(input_dir, f"{test_theme}_{object_class}_seed{seed}.jpg")
                        image = Image.open(img_path)
                        target_image = image_transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            res = model(target_image)
                            label = torch.tensor([theme_label]).to(device)
                            loss = torch.nn.functional.cross_entropy(res, label)
                            # softmax the prediction
                            res_softmax = torch.nn.functional.softmax(res, dim=1)
                            pred_loss = res_softmax[0][theme_label]
                            pred_label = torch.argmax(res)
                            pred_success = (torch.argmax(res) == theme_label).sum()

                        results["loss"][test_theme] += loss
                        results["pred_loss"][test_theme] += pred_loss
                        results["acc"][test_theme] += (pred_success * 1.0 / (len(class_available) * len(args.seed)))

                        misclassified_as = theme_available[pred_label.item()]
                        results["misclassified"][test_theme][misclassified_as] += 1

            if not self.config['dry_run']:
                torch.save(results, output_path)

        else:
            for test_theme in tqdm(theme_available):
                for seed in seed:
                    for idx, object_class in enumerate(class_available):
                        theme_label = idx
                        img_path = os.path.join(input_dir, f"{test_theme}_{object_class}_seed{seed}.jpg")
                        image = Image.open(img_path)
                        target_image = image_transform(image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            res = model(target_image)
                            label = torch.tensor([theme_label]).to(device)
                            loss = torch.nn.functional.cross_entropy(res, label)
                            # softmax the prediction
                            res_softmax = torch.nn.functional.softmax(res, dim=1)
                            pred_loss = res_softmax[0][theme_label]
                            pred_success = (torch.argmax(res) == theme_label).sum()
                            pred_label = torch.argmax(res)

                        results["loss"][object_class] += loss
                        results["pred_loss"][object_class] += pred_loss
                        results["acc"][object_class] += (pred_success * 1.0 / (len(theme_available) * len(seed)))
                        misclassified_as = class_available[pred_label.item()]
                        results["misclassified"][object_class][misclassified_as] += 1

            if not self.config['dry_run']:
                torch.save(results, output_path)

        self.results.update(results)

    def calculate_fid_score(self):
        """Calculate FID score using the utilities from utils.py."""
        generated_images = load_style_generated_images(self.config['output_dir'], self.config['theme'])
        reference_images = load_style_ref_images(self.config['original_image_dir'], self.config['theme'])

        fid_score = calculate_fid(generated_images, reference_images, batch_size=self.config['batch_size'])
        self.results["FID"] = fid_score
        self.logger.info(f"FID Score calculated: {fid_score}")

        self.save_results(self.results["FID"])

    def save_results(self,result):
        """Save the results."""
        output_path = os.path.join(self.config['eval_output_dir'], "evaluation_results.pth")
        torch.save(result, output_path)
        self.logger.info(f"Results saved to: {output_path}")

    def run(self):
        """Run the full pipeline: image generation, accuracy, and FID."""
        self.logger.info("Starting the evaluation pipeline...")
        self.load_model()
        self.generator.sample_image()
        self.calculate_accuracy()
        self.calculate_fid_score()
        # self.save_results()
        self.logger.info("Evaluation completed successfully.")


def main():
    """Main entry point for running the entire pipeline."""
    parser = ArgumentParser(description="Unified ESD Evaluation and Sampling")
    parser.add_argument('--config_path', required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_path)

    # Initialize and run the evaluation
    evaluator = ESDEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
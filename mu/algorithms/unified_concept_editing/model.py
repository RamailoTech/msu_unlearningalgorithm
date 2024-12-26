# mu/algorithms/unified_concept_editing/model.py
import numpy as np
from mu.algorithms.unified_concept_editing.utils import get_ratios
import torch
from typing import Any, List, Optional
import logging
import copy
from tqdm import tqdm
import ast
from diffusers import StableDiffusionPipeline


from mu.core import BaseModel

class UnifiedConceptEditingModel(BaseModel):
    """
    UnifiedConceptEditingModel handles loading, saving, and interacting with the Stable Diffusion model using diffusers.
    """

    def __init__(self, ckpt_path: str, device: str):
        """
        Initialize the UnifiedConceptEditingModel.

        Args:
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on (e.g., 'cuda').
        """
        super().__init__()
        self.device = device
        self.ckpt_path = ckpt_path
        self.model = self.load_model(ckpt_path, device)
        self.unet = self.model.unet  # Expose UNet for editing
        self.logger = logging.getLogger(__name__)


    def load_model(self, ckpt_path: str, device: str) -> StableDiffusionPipeline:
        """
        Load the Stable Diffusion model from the checkpoint.

        Args:
            ckpt_path (str): Path to the model checkpoint.
            device (str): Device to load the model on.

        Returns:
            StableDiffusionPipeline: Loaded Stable Diffusion model.
        """
        model = StableDiffusionPipeline.from_pretrained(
            ckpt_path,
            torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32
        ).to(device)
        model.enable_attention_slicing()  # Optimize memory usage
        return model

    def save_model(self, model, output_path: str):
        """
        Save the model's state dictionary.

        Args:
            output_path (str): Path to save the model checkpoint.
        """
        self.logger.info(f"Saving model to {output_path}...")
        model.save_pretrained(output_path)
        self.logger.info("Model saved successfully.")

    def edit_model_erase(
        self,
        old_texts: List[str],
        new_texts: List[str],
        retain_texts: List[str],
        lamb: float = 0.5,
        erase_scale: float = 1.0,
        preserve_scale: float = 0.1,
        layers_to_edit: Optional[List[int]] = None,
        technique: str = 'replace'
    ):
        """
        Edit the model by modifying cross-attention layers to erase or replace concepts.

        Args:
            old_texts (List[str]): List of old concepts to erase.
            new_texts (List[str]): List of new concepts to replace with.
            retain_texts (List[str]): List of concepts to retain.
            lamb (float, optional): Lambda parameter for loss. Defaults to 0.5.
            erase_scale (float, optional): Scale for erasing concepts. Defaults to 1.0.
            preserve_scale (float, optional): Scale for preserving concepts. Defaults to 0.1.
            layers_to_edit (Optional[List[int]], optional): Specific layers to edit. Defaults to None.
            technique (str, optional): Technique to erase ('replace' or 'tensor'). Defaults to 'replace'.

        Returns:
            StableDiffusionPipeline: Edited Stable Diffusion model.
        """
        sub_nets = self.unet.named_children()
        ca_layers = []
        for net in sub_nets:
            if 'up' in net[0] or 'down' in net[0]:
                for block in net[1]:
                    if 'Cross' in block.__class__.__name__:
                        for attn in block.attentions:
                            for transformer in attn.transformer_blocks:
                                ca_layers.append(transformer.attn2)
            if 'mid' in net[0]:
                for attn in net[1].attentions:
                    for transformer in attn.transformer_blocks:
                        ca_layers.append(transformer.attn2)

        # Get value and key modules
        projection_matrices = [l.to_v for l in ca_layers]
        og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
        if True:  # Assuming 'with_to_k' is always True
            projection_matrices += [l.to_k for l in ca_layers]
            og_matrices += [copy.deepcopy(l.to_k) for l in ca_layers]

        # Reset parameters
        num_ca_clip_layers = len(ca_layers)
        for idx, l in enumerate(ca_layers):
            l.to_v = copy.deepcopy(og_matrices[idx])
            projection_matrices[idx] = l.to_v
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx])
            projection_matrices[num_ca_clip_layers + idx] = l.to_k

        # Convert layers_to_edit from string to list if necessary
        if isinstance(layers_to_edit, str):
            layers_to_edit = ast.literal_eval(layers_to_edit)

        # Begin editing
        for layer_num in tqdm(range(len(projection_matrices)), desc="Editing Layers"):
            if layers_to_edit is not None and layer_num not in layers_to_edit:
                continue

            with torch.autocast(self.device):
                with torch.no_grad():
                    # Initialize matrices
                    mat1 = lamb * projection_matrices[layer_num].weight
                    mat2 = lamb * torch.eye(
                        projection_matrices[layer_num].weight.shape[1],
                        device=projection_matrices[layer_num].weight.device
                    )

                    # Iterate over old and new texts to compute modifications
                    for old_text, new_text in zip(old_texts, new_texts):
                        texts = [old_text, new_text]
                        text_input = self.model.tokenizer(
                            texts,
                            padding="max_length",
                            max_length=self.model.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]

                        # Determine token indices
                        final_token_idx = text_input.attention_mask[0].sum().item() - 2
                        final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                        farthest = max(final_token_idx_new, final_token_idx)

                        # Extract embeddings
                        old_emb = text_embeddings[0, final_token_idx : len(text_embeddings[0]) - max(0, farthest - final_token_idx)]
                        new_emb = text_embeddings[1, final_token_idx_new : len(text_embeddings[1]) - max(0, farthest - final_token_idx_new)]

                        context = old_emb.detach()

                        values = []
                        with torch.no_grad():
                            for layer in projection_matrices:
                                if technique == 'tensor':
                                    o_embs = layer(old_emb).detach()
                                    u = o_embs / o_embs.norm()

                                    new_embs = layer(new_emb).detach()
                                    new_emb_proj = (u * new_embs).sum()

                                    target = new_embs - (new_emb_proj) * u
                                    values.append(target.detach())
                                elif technique == 'replace':
                                    values.append(layer(new_emb).detach())
                                else:
                                    values.append(layer(new_emb).detach())

                        # Compute context and value vectors
                        context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                        context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                        value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)

                        # Update mat1 and mat2
                        for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                        for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                        mat1 += erase_scale * for_mat1
                        mat2 += erase_scale * for_mat2

                    # Handle retain_texts to preserve certain concepts
                    for old_text, new_text in zip(retain_texts, retain_texts):
                        texts = [old_text, new_text]
                        text_input = self.model.tokenizer(
                            texts,
                            padding="max_length",
                            max_length=self.model.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
                        old_emb, new_emb = text_embeddings
                        context = old_emb.detach()

                        values = []
                        with torch.no_grad():
                            for layer in projection_matrices:
                                values.append(layer(new_emb).detach())

                        context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                        context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                        value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)

                        for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                        for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                        if preserve_scale is None:
                            preserve_scale = max(0.1, 1 / len(retain_texts))
                        mat1 += preserve_scale * for_mat1
                        mat2 += preserve_scale * for_mat2

                        # Update projection matrix
                        projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

        return self.model

    def edit_model_debias(
        self,
        old_text_,
        new_text_,
        retain_text_,
        add=True,
        layers_to_edit=None,
        lamb=0.1,
        erase_scale=0.1,
        preserve_scale=0.1,
        with_to_k=True,
        num_images=1
    ):
        """
        Edit the model to debias certain concepts.

        Args:
            self.model: The stable diffusion model.
            old_text_ (List[str]): List of old concepts to debias.
            new_text_ (List[List[str]]): List of new concepts to replace with.
            retain_text_ (List[str]): List of concepts to retain.
            add (bool, optional): Whether to add old text to new text. Defaults to True.
            layers_to_edit (Optional[List[int]], optional): Specific layers to edit. Defaults to None.
            lamb (float, optional): Lambda parameter for loss. Defaults to 0.1.
            erase_scale (float, optional): Scale for erasing concepts. Defaults to 0.1.
            preserve_scale (float, optional): Scale for preserving concepts. Defaults to 0.1.
            with_to_k (bool, optional): Whether to include key projection matrices. Defaults to True.
            num_images (int, optional): Number of images to generate for ratio calculation. Defaults to 1.

        Returns:
            Tuple: Edited model, weights, initial ratios, final ratios.
        """
        max_bias_diff = 0.05
        sub_nets = self.unet.named_children()
        ca_layers = []
        for net in sub_nets:
            if 'up' in net[0] or 'down' in net[0]:
                for block in net[1]:
                    if 'Cross' in block.__class__.__name__:
                        for attn in block.attentions:
                            for transformer in attn.transformer_blocks:
                                ca_layers.append(transformer.attn2)
            if 'mid' in net[0]:
                for attn in net[1].attentions:
                    for transformer in attn.transformer_blocks:
                        ca_layers.append(transformer.attn2)

        projection_matrices = [l.to_v for l in ca_layers]
        og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
        if with_to_k:
            projection_matrices += [l.to_k for l in ca_layers]
            og_matrices += [copy.deepcopy(l.to_k) for l in ca_layers]

        num_ca_clip_layers = len(ca_layers)
        for idx_, l in enumerate(ca_layers):
            l.to_v = copy.deepcopy(og_matrices[idx_])
            projection_matrices[idx_] = l.to_v
            if with_to_k:
                l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
                projection_matrices[num_ca_clip_layers + idx_] = l.to_k

        layers_to_edit = ast.literal_eval(layers_to_edit) if isinstance(layers_to_edit, str) else layers_to_edit
        lamb = ast.literal_eval(lamb) if isinstance(lamb, str) else lamb

        old_texts = []
        new_texts = []
        for old_text, new_text in zip(old_text_, new_text_):
            old_texts.append(old_text)
            n_t = []
            for t in new_text:
                if (old_text.lower() not in t.lower()) and add:
                    n_t.append(t + ' ' + old_text)
                else:
                    n_t.append(t)
            if len(n_t) == 1:
                n_t *= 2
            new_texts.append(n_t)
        if retain_text_ is None:
            ret_texts = ['']
            retain = False
        else:
            ret_texts = retain_text_
            retain = True

        self.logger.info(f"Old texts: {old_texts}, New texts: {new_texts}")
        desired_ratios = [torch.ones(len(c)) / len(c) for c in new_texts]
        weight_step = 0.1
        weights = [torch.zeros(len(c)) for c in new_texts]

        for i in range(30):
            max_ratio_gap = max_bias_diff
            if i == 0:
                prev_ratio = None
                ratio_diff = None
            else:
                prev_ratio = ratios
                ratio_diff = max_change
            ratios = [0 for _ in desired_ratios]
            ratios = get_ratios(
                ldm_stable=self.model,
                prev_ratio=prev_ratio,
                ratio_diff=ratio_diff,
                max_ratio_gap=max_ratio_gap,
                concepts=old_texts,
                classes=new_texts,
                num_samples=num_images
            )
            if i == 0:
                init_ratios = ratios
            self.logger.info(f"Ratios: {ratios}")
            max_change = [(ratio - desired_ratio).abs().max() for ratio, desired_ratio in zip(ratios, desired_ratios)]

            if max(max_change) < max_bias_diff:
                self.logger.info(f"All concepts are debiased at Iteration: {i}")
                break

            weights_delta = [weight_step * (desired_ratio - ratio) for ratio, desired_ratio in zip(ratios, desired_ratios)]
            weights_delta = [weights_delta[idx] if max_c > max_bias_diff else weights_delta[idx] * 0 for idx, max_c in enumerate(max_change)]

            ret_text_add = [old_texts[idx] for idx, weight in enumerate(weights_delta) if weight[0] == 0]
            if len(ret_text_add) > 0:
                ret_texts += ret_text_add
                ret_texts = list(np.unique(ret_texts))
            weights = weights_delta

            for layer_num in range(len(projection_matrices)):
                if layers_to_edit is not None and layer_num not in layers_to_edit:
                    continue

                with torch.no_grad():
                    mat1 = lamb * projection_matrices[layer_num].weight
                    mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device=projection_matrices[layer_num].weight.device)

                    for cnt, t in enumerate(zip(old_texts, new_texts)):
                        old_text = t[0]
                        new_text = t[1]
                        texts = [old_text] + new_text
                        text_input = self.model.tokenizer(
                            texts,
                            padding="max_length",
                            max_length=self.model.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
                        old_emb = text_embeddings[0]
                        final_token_idx = text_input.attention_mask[0].sum().item() - 2
                        final_token_idx_new = [text_input.attention_mask[i].sum().item() - 2 for i in range(1, len(text_input.attention_mask))]
                        farthest = max(final_token_idx_new + [final_token_idx])
                        new_emb = text_embeddings[1:]

                        context = old_emb.detach()[final_token_idx:len(old_emb) - max(0, farthest - final_token_idx)]
                        values = []
                        with torch.no_grad():
                            for layer in projection_matrices:
                                o_embs = layer(old_emb).detach()
                                o_embs = o_embs[final_token_idx:len(o_embs) - max(0, farthest - final_token_idx)]
                                embs = layer(new_emb[:]).detach()
                                target = o_embs
                                for j, emb in enumerate(embs):
                                    u = emb
                                    u = u[final_token_idx_new[j]:len(u) - max(0, farthest - final_token_idx_new[j])]
                                    u = u / u.norm()
                                    o_emb_proj = (u * o_embs).sum()
                                    target += (weights[cnt][j] * o_embs.norm()) * u
                                values.append(target.detach())
                        context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                        context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                        value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                        for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                        for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                        mat1 += erase_scale * for_mat1
                        mat2 += erase_scale * for_mat2

                    for old_text, new_text in zip(ret_texts, ret_texts):
                        text_input = self.model.tokenizer(
                            [old_text, new_text],
                            padding="max_length",
                            max_length=self.model.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
                        old_emb, new_emb = text_embeddings
                        context = old_emb.detach()
                        values = []
                        with torch.no_grad():
                            for layer in projection_matrices:
                                values.append(layer(new_emb[:]).detach())
                        context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                        context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                        value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                        for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                        for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                        mat1 += preserve_scale * for_mat1
                        mat2 += preserve_scale * for_mat2
                    projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

        self.logger.info(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
        self.logger.info(f'Initial Ratios: {init_ratios} Final Ratios: {ratios} and Weights: {weights}')
        return self.model, weights, init_ratios, ratios
    
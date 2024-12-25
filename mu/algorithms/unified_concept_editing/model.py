# mu/algorithms/unified_concept_editing/model.py

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

    def edit_model(
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

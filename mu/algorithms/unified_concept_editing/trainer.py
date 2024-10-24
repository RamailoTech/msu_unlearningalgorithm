# unified_concept_trainer.py

from base_trainer import BaseTrainer
import torch
import copy
import ast
from tqdm import tqdm

class UnifiedConceptTrainer(BaseTrainer):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.with_to_k = config.get('with_to_k', True)
        self.lamb = config.get('lamb', 0.1)
        self.erase_scale = config.get('erase_scale', 0.1)
        self.preserve_scale = config.get('preserve_scale', 0.1)
        self.technique = config.get('technique', 'replace')
        self.layers_to_edit = config.get('layers_to_edit', None)
        self.device = config.get('device', 'cuda')

    def train(self, old_texts, new_texts, retain_texts, approach='erasing'):
        if approach == 'erasing':
            self.erasing_approach(old_texts, new_texts, retain_texts)
        elif approach == 'debiasing':
            self.debiasing_approach(old_texts, new_texts, retain_texts)
        else:
            raise ValueError("Approach must be either 'erasing' or 'debiasing'")

    def erasing_approach(self, old_texts, new_texts, retain_texts):
        ldm_stable = self.model.pipeline
        device = self.device

        # Collect cross-attention layers
        ca_layers = self._collect_cross_attention_layers(ldm_stable)

        # Get projection matrices
        projection_matrices, original_matrices = self._get_projection_matrices(ca_layers)

        # Reset parameters
        self._reset_projection_matrices(projection_matrices, original_matrices)

        # Main editing loop
        for layer_num, layer in enumerate(projection_matrices):
            if self.layers_to_edit and layer_num not in self.layers_to_edit:
                continue

            mat1 = self.lamb * layer.weight
            mat2 = self.lamb * torch.eye(layer.weight.shape[1], device=device)

            for old_text, new_text in zip(old_texts, new_texts):
                mat1, mat2 = self._update_matrices(ldm_stable, old_text, new_text, layer_num, mat1, mat2, erase=True)

            for retain_text in retain_texts:
                mat1, mat2 = self._update_matrices(ldm_stable, retain_text, retain_text, layer_num, mat1, mat2, erase=False)

            # Update projection matrix
            layer.weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    def debiasing_approach(self, old_texts, new_texts_list, retain_texts):
        ldm_stable = self.model.pipeline
        device = self.device

        # Collect cross-attention layers
        ca_layers = self._collect_cross_attention_layers(ldm_stable)

        # Get projection matrices
        projection_matrices, original_matrices = self._get_projection_matrices(ca_layers)

        # Reset parameters
        self._reset_projection_matrices(projection_matrices, original_matrices)

        # Initialize variables
        max_iterations = 30
        weight_step = 0.1
        max_bias_diff = 0.05  # Hyperparameter for acceptable bias difference

        # Debiasing loop
        for iteration in range(max_iterations):
            ratios = self._compute_bias_ratios(ldm_stable, old_texts, new_texts_list)
            max_change = [abs(ratio - 1.0 / len(new_texts)).max() for ratio, new_texts in zip(ratios, new_texts_list)]

            if max(max_change) < max_bias_diff:
                print(f'All concepts are debiased at Iteration: {iteration}')
                break

            weights_delta = [weight_step * (1.0 / len(new_texts) - ratio) for ratio, new_texts in zip(ratios, new_texts_list)]
            weights = [w + delta for w, delta in zip([torch.zeros(len(nt)) for nt in new_texts_list], weights_delta)]

            # Update retain texts if needed
            for idx, change in enumerate(max_change):
                if change < max_bias_diff:
                    retain_texts.append(old_texts[idx])

            # Main editing loop per iteration
            for layer_num, layer in enumerate(projection_matrices):
                if self.layers_to_edit and layer_num not in self.layers_to_edit:
                    continue

                mat1 = self.lamb * layer.weight
                mat2 = self.lamb * torch.eye(layer.weight.shape[1], device=device)

                for idx, old_text in enumerate(old_texts):
                    new_texts = new_texts_list[idx]
                    weights_current = weights[idx]
                    mat1, mat2 = self._update_matrices_debias(ldm_stable, old_text, new_texts, weights_current, layer_num, mat1, mat2)

                for retain_text in retain_texts:
                    mat1, mat2 = self._update_matrices(ldm_stable, retain_text, retain_text, layer_num, mat1, mat2, erase=False)

                # Update projection matrix
                layer.weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    def _collect_cross_attention_layers(self, ldm_stable):
        ca_layers = []
        for module in ldm_stable.unet.modules():
            if module.__class__.__name__ == 'CrossAttention':
                ca_layers.append(module)
        return ca_layers

    def _get_projection_matrices(self, ca_layers):
        projection_matrices = [layer.to_v for layer in ca_layers]
        original_matrices = [copy.deepcopy(layer.to_v) for layer in ca_layers]
        if self.with_to_k:
            projection_matrices.extend([layer.to_k for layer in ca_layers])
            original_matrices.extend([copy.deepcopy(layer.to_k) for layer in ca_layers])
        return projection_matrices, original_matrices

    def _reset_projection_matrices(self, projection_matrices, original_matrices):
        for idx, layer in enumerate(projection_matrices):
            layer.weight.data = original_matrices[idx].weight.data.clone()

    def _update_matrices(self, ldm_stable, old_text, new_text, layer_num, mat1, mat2, erase=True):
        device = self.device
        texts = [old_text, new_text]
        text_inputs = ldm_stable.tokenizer(
            texts,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = ldm_stable.text_encoder(text_inputs.input_ids.to(device))[0]

        old_emb = text_embeddings[0]
        new_emb = text_embeddings[1]

        context = old_emb.detach()
        values = [layer(new_emb).detach() for layer in self._get_projection_matrices(self._collect_cross_attention_layers(ldm_stable))[0]]

        context_vector = context.unsqueeze(-1)
        context_vector_T = context.unsqueeze(-2)
        value_vector = values[layer_num].unsqueeze(-1)

        scale = self.erase_scale if erase else self.preserve_scale

        for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
        for_mat2 = (context_vector @ context_vector_T).sum(dim=0)

        mat1 += scale * for_mat1
        mat2 += scale * for_mat2

        return mat1, mat2

    def _update_matrices_debias(self, ldm_stable, old_text, new_texts, weights_current, layer_num, mat1, mat2):
        device = self.device
        texts = [old_text] + new_texts
        text_inputs = ldm_stable.tokenizer(
            texts,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = ldm_stable.text_encoder(text_inputs.input_ids.to(device))[0]

        old_emb = text_embeddings[0]
        new_embs = text_embeddings[1:]

        context = old_emb.detach()
        values = [layer(new_emb).detach() for new_emb in new_embs for layer in self._get_projection_matrices(self._collect_cross_attention_layers(ldm_stable))[0]]

        context_vector = context.unsqueeze(-1)
        context_vector_T = context.unsqueeze(-2)

        # Sum over new_texts with corresponding weights
        for value_vector, weight in zip(values, weights_current):
            value_vector = value_vector.unsqueeze(-1)
            for_mat1 = weight * (value_vector @ context_vector_T).sum(dim=0)
            for_mat2 = weight * (context_vector @ context_vector_T).sum(dim=0)
            mat1 += self.erase_scale * for_mat1
            mat2 += self.erase_scale * for_mat2

        return mat1, mat2

    def _compute_bias_ratios(self, ldm_stable, old_texts, new_texts_list):
        # Placeholder function to compute bias ratios
        # Implement the actual logic based on the original code
        ratios = []
        for old_text, new_texts in zip(old_texts, new_texts_list):
            # Generate images and compute ratios using CLIP model
            # For simplicity, we'll assume uniform ratios here
            ratios.append(torch.tensor([1.0 / len(new_texts) for _ in new_texts]))
        return ratios

    def compute_loss(self, output: Any, target: Any) -> Any:
        pass  # Not used

    def step_optimizer(self):
        pass  # Not used

    def validate(self, *args, **kwargs):
        pass  # Not used

    def save_checkpoint(self, output_path: str):
        self.model.save_model(output_path)

    def get_model_params(self) -> Any:
        return self.model.state_dict()

    def set_model_params(self, params: Any):
        self.model.load_state_dict(params)

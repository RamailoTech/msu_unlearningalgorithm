import torch 
from typing import Any


class AttnController:
    def __init__(self) -> None:
        self.attn_probs = []
        self.logs = []

    def __call__(self, attn_prob, m_name) -> Any:
        bs, _ = self.concept_positions.shape
        head_num = attn_prob.shape[0] // bs
        target_attns = attn_prob.masked_select(self.concept_positions[:, None, :].repeat(head_num, 1, 1)).reshape(
            -1, self.concept_positions[0].sum())
        self.attn_probs.append(target_attns)
        self.logs.append(m_name)

    def set_concept_positions(self, concept_positions):
        self.concept_positions = concept_positions

    def loss(self):
        return torch.cat(self.attn_probs).norm()

    def zero_attn_probs(self):
        self.attn_probs = []
        self.logs = []
        self.concept_positions = None


class MyCrossAttnProcessor:
    def __init__(self, attn_controller: "AttnController", module_name) -> None:
        self.attn_controller = attn_controller
        self.module_name = module_name

    def __call__(self, attn: "CrossAttention", hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attn_controller(attention_probs, self.module_name)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
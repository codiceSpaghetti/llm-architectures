import re
from typing import Any

from .args import Gemma3Config


class Gemma3Converter:
    def __init__(self, config: Gemma3Config):
        self.config = config

        self.from_hf_map = {
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            "model.language_model.norm.weight": "lm_norm.weight",
            "lm_head.weight": "lm_head.weight",
            "model.language_model.layers.{}.input_layernorm.weight": (
                "layers.{}.pre_attention_norm.weight"
            ),
            "model.language_model.layers.{}.post_attention_layernorm.weight": (
                "layers.{}.post_attention_norm.weight"
            ),
            "model.language_model.layers.{}.pre_feedforward_layernorm.weight": (
                "layers.{}.pre_ffn_norm.weight"
            ),
            "model.language_model.layers.{}.post_feedforward_layernorm.weight": (
                "layers.{}.post_ffn_norm.weight"
            ),
            "model.language_model.layers.{}.self_attn.q_proj.weight": (
                "layers.{}.attention.wq.weight"
            ),
            "model.language_model.layers.{}.self_attn.k_proj.weight": (
                "layers.{}.attention.wk.weight"
            ),
            "model.language_model.layers.{}.self_attn.v_proj.weight": (
                "layers.{}.attention.wv.weight"
            ),
            "model.language_model.layers.{}.self_attn.o_proj.weight": (
                "layers.{}.attention.wo.weight"
            ),
            "model.language_model.layers.{}.self_attn.q_norm.weight": (
                "layers.{}.attention.q_norm.weight"
            ),
            "model.language_model.layers.{}.self_attn.k_norm.weight": (
                "layers.{}.attention.k_norm.weight"
            ),
            "model.language_model.layers.{}.mlp.gate_proj.weight": (
                "layers.{}.feed_forward.w1.weight"
            ),
            "model.language_model.layers.{}.mlp.up_proj.weight": (
                "layers.{}.feed_forward.w3.weight"
            ),
            "model.language_model.layers.{}.mlp.down_proj.weight": (
                "layers.{}.feed_forward.w2.weight"
            ),
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        text_model_state_dict = {}
        for key, value in hf_state_dict.items():
            if key.startswith("model.language_model") or key.startswith("lm_head"):
                text_model_state_dict[key] = value

        state_dict = {}

        for key, value in text_model_state_dict.items():
            if "layers" in key:
                match = re.search(r"\d+", key)
                layer_num = match.group(0) if match else None
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in self.from_hf_map:
                new_key = self.from_hf_map[key]
                if layer_num:
                    new_key = new_key.format(layer_num)
                state_dict[new_key] = value

        return state_dict

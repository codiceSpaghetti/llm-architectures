import re
from typing import Any

from .args import Gemma3Config

class Gemma3Converter:
    def __init__(self, config: Gemma3Config):
        self.config = config

        self.from_hf_map = {
            # Embeddings
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            
            # Final norm and head
            "model.language_model.norm.weight": "lm_norm.weight", 
            "lm_head.weight": "lm_head.weight",
            
            # Layer norm mappings
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.pre_attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
            "model.language_model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.pre_ffn_norm.weight",  
            "model.language_model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
            
            # Attention projections (for layers with full attention)
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight", 
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            
            # Query/Key normalization (for layers with attention)
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            
            # MLP projections (for layers with MLP)
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight", 
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                layer_num = re.search(r"\d+", key).group(0)
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in self.from_hf_map:
                new_key = self.from_hf_map[key]
                if new_key is None:
                    print(f"Skipping {key}")
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)
                print(f"Mapping {key} to {new_key}")
                state_dict[new_key] = value
            
        return state_dict

    def load_hf_model(self, model_name: str = "google/gemma-3-4b-it"):
        """Load Hugging Face model and convert to our format"""
        from transformers import AutoModelForCausalLM
        import torch
        
        # Load the HF model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.bfloat16,
            device_map="cpu"
        )
        
        # Get only the language model state dict (skip vision components)
        hf_state_dict = {}
        for key, value in hf_model.state_dict().items():
            if key.startswith("model.language_model") or key.startswith("lm_head"):
                hf_state_dict[key] = value
        
        # Convert to our format
        our_state_dict = self.from_hf(hf_state_dict)
        
        return our_state_dict
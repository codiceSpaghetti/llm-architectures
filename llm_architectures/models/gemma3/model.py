from typing import Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_architectures.models.gemma3.args import Gemma3Config
from llm_architectures.models.gemma3.args import gemma3_configs
from llm_architectures.models.utils import get_torch_dtype


class AttentionType(Enum):
    GLOBAL = "global"
    LOCAL = "local"


class RoPE:
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0, rope_scaling_factor: int = 1) -> None:
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        freqs = freqs/rope_scaling_factor
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    def __call__(self, xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [batch, heads, seq_len, head_dim]
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        seq_len = xq_.shape[2]
        freqs_cis = self.freqs_cis[0:seq_len].view(1, 1, seq_len, self.freqs_cis.shape[-1])
        
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    
def create_attention_mask(seq_len: int, window_size: Union[int, None] = None) -> torch.Tensor:
    # Start with lower triangular matrix (causal mask)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    
    # Zero out elements beyond the window size
    # This creates the sliding window by removing distant past tokens
    if window_size is not None and window_size < seq_len:
        mask = mask & torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 
                                diagonal=-(window_size - 1))
    return mask.float()


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.attention_type = config.attention_type
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        # Create linear layers with default dtype, will be converted during weight loading
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.bias)
        
        self.query_pre_attn_scalar = config.query_pre_attn_scalar
        self.use_qk_norm = config.use_qk_norm
        
        self.q_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if self.use_qk_norm
            else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if self.use_qk_norm
            else None
        )
        
        if self.query_pre_attn_scalar is not None:
            self.scaling = self.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5
        
        if self.attention_type == AttentionType.LOCAL:
            mask = create_attention_mask(config.block_size, config.sliding_window)
            self.rope = RoPE(self.head_dim, config.block_size, config.rope_theta_local, config.rope_scaling_factor_local)
        else:
            mask = create_attention_mask(config.block_size, None)
            self.rope = RoPE(self.head_dim, config.block_size, config.rope_theta_global, config.rope_scaling_factor_global)
        
        self.register_buffer("mask", mask, persistent=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        q, k = self.rope(q, k)
        
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # Apply causal mask
        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1)
        
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        
        return self.wo(output)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class GeGLU(nn.Module):
    """GeGLU implementation using GELU activation function."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


class Gemma3Block(torch.nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        self.pre_attention_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.attention = Gemma3Attention(config)
        self.post_attention_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.pre_ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.feed_forward = GeGLU(config.dim, config.intermediate_size, bias=config.bias)
        self.post_ffn_norm = RMSNorm(config.dim, config.rms_norm_eps)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_norm(self.attention(self.pre_attention_norm(x)))
        x = x + self.post_ffn_norm(self.feed_forward(self.pre_ffn_norm(x)))
        return x


class Gemma3(torch.nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config

        # Create with default dtype, will be converted during weight loading
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.normalizer = torch.tensor(config.dim**0.5)

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            if i % 5 == 0 and i != 0:
                config.attention_type = AttentionType.GLOBAL
            else:
                config.attention_type = AttentionType.LOCAL

            self.layers.append(Gemma3Block(config))

        self.lm_norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(x)
        x = x * self.normalizer
        for layer in self.layers:
            x = layer(x)
        x = self.lm_norm(x)
        x = self.lm_head(x)
        return x


if __name__ == "__main__":
    breakpoint()
    config = Gemma3Config(gemma3_configs["4B"])
    model = Gemma3(config)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", dtype=get_torch_dtype(config.dtype), device_map="cpu", trust_remote_code=True)
    hf_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    for i in range(len(model.layers)):
        breakpoint()
        model_output = model.layers[i](inputs)
        hf_output = hf_model.layers[i](inputs)
        
        print(f"Layer {i} output:")
        print(model_output)
        print(hf_output)
        print("-" * 100)
    
    print(model)
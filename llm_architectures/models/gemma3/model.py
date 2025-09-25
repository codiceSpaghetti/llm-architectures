"""
GEMMA3 IDIOSYNCRASIES (llama as baseline)
* Alternate global and local attention every 6th layer to save memory and compute
* Utilize tanh approximation for GeGLU instead of SwiGLU
* Unsure about pre or post normalization in attention blocks? They did both!
* Then why not using also query and key normalization?
* normalizer embedding multiplying by sqrt(dim) for RMSNorm
* Different base RoPE frequencies for global and local attention
* Rope scaling factor only for global attention
* RMSNorm multiplies the output by (1 + w), not just w
* Instead of normalizing softmax for sqrt(head_dim), specify a different normalizer (just for 27B variant)

THE DEVIL IS IN THE DETAILS (which caused me headaches)
* RoPE must be done in float32 and not float16
* RMSNorm multiplies the output by (1 + w) and not just w
  (this was a silly oversight on my part!)
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_architectures.models.gemma3.args import AttentionType, Gemma3Config
from llm_architectures.models.utils import get_torch_dtype


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_scaling_factor: int = 1) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].to(torch.float32) / dim))
    freqs = freqs / rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).to(torch.float32)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).to(torch.float32), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out


def create_attention_mask(
    seq_len: int,
    window_size: Union[int, None] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # Start with lower triangular matrix (causal mask)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    # Zero out elements beyond the window size
    # This creates the sliding window by removing distant past tokens
    if window_size is not None and window_size < seq_len:
        mask = mask & torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-(window_size - 1))
    return mask.to(dtype)


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.attention_type = config.attention_type
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        dtype = get_torch_dtype(config.dtype)
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.bias, dtype=dtype)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.bias, dtype=dtype)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.bias, dtype=dtype)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.bias, dtype=dtype)

        self.query_pre_attn_scalar = config.query_pre_attn_scalar

        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # it differs from head_dim**-0.5 only for the 27B variant
        if self.query_pre_attn_scalar is not None:
            self.scaling = self.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        if self.attention_type == AttentionType.LOCAL:
            mask = create_attention_mask(config.block_size, config.sliding_window, dtype)
        else:
            mask = create_attention_mask(config.block_size, None, dtype)

        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        q = q * self.scaling
        scores = torch.matmul(q, k.transpose(-2, -1))

        mask = self.mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = F.softmax(scores, dim=-1)

        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)

        return self.wo(output)


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.to(torch.float32))
        output = output * (1 + self.weight.to(torch.float32))
        return output.type_as(x)


class GeGLU(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias, dtype=dtype)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias, dtype=dtype)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x), approximate="tanh") * self.w3(x))


class Gemma3Block(torch.nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        dtype = get_torch_dtype(config.dtype)

        self.pre_attention_norm = Gemma3RMSNorm(config.dim, config.rms_norm_eps)
        self.attention = Gemma3Attention(config)
        self.post_attention_norm = Gemma3RMSNorm(config.dim, config.rms_norm_eps)
        self.pre_ffn_norm = Gemma3RMSNorm(config.dim, config.rms_norm_eps)
        self.feed_forward = GeGLU(config.dim, config.intermediate_size, bias=config.bias, dtype=dtype)
        self.post_ffn_norm = Gemma3RMSNorm(config.dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_norm(self.attention(self.pre_attention_norm(x), freqs_cis))
        x = x + self.post_ffn_norm(self.feed_forward(self.pre_ffn_norm(x)))
        return x


class Gemma3(torch.nn.Module):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config
        dtype = get_torch_dtype(config.dtype)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, dtype=dtype)

        self.register_buffer("normalizer", torch.tensor(config.dim**0.5, dtype=dtype), persistent=False)

        self.register_buffer(
            "local_freqs_cis",
            precompute_freqs_cis(
                config.head_dim,
                config.block_size * 2,
                theta=config.rope_theta_local,
                rope_scaling_factor=config.rope_scaling_factor_local,
            ),
            persistent=False,
        )
        self.register_buffer(
            "global_freqs_cis",
            precompute_freqs_cis(
                config.head_dim,
                config.block_size * 2,
                theta=config.rope_theta_global,
                rope_scaling_factor=config.rope_scaling_factor_global,
            ),
            persistent=False,
        )

        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            # Every 6th layer is full_attention, others are sliding_attention
            if (i + 1) % 6 == 0:
                config.attention_type = AttentionType.GLOBAL
            else:
                config.attention_type = AttentionType.LOCAL

            self.layers.append(Gemma3Block(config))

        self.lm_norm = Gemma3RMSNorm(config.dim, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False, dtype=dtype)
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.tok_embeddings(x)
        x = x * self.normalizer

        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

        for i, layer in enumerate(self.layers):
            # Determine which freqs_cis to use based on layer attention type
            if (i + 1) % 6 == 0:  # Global attention
                freqs_cis = self.global_freqs_cis.index_select(0, positions)
            else:  # Local attention
                freqs_cis = self.local_freqs_cis.index_select(0, positions)

            x = layer(x, freqs_cis)

        x = self.lm_norm(x)
        x = self.lm_head(x)
        return x

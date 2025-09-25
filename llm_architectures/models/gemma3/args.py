from dataclasses import dataclass


@dataclass
class Gemma3Config:
    dtype: str = "bfloat16"
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    dim: int = 4096
    intermediate_size: int = 14336
    head_dim: int = 256
    sliding_window: int = 1024
    vocab_size: int = 128256
    use_qk_norm: bool = True
    rope_theta_global: int = 1_000_000
    rope_theta_local: int = 10_000
    rope_scaling_factor_global: int = 8
    rope_scaling_factor_local: int = 1
    block_size: int = 8192
    bias: bool = False
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float | None = None
    

gemma3_configs = {
    "4B": Gemma3Config(
        n_layers=34,
        n_heads=8,
        n_kv_heads=4,
        dim=2560,
        intermediate_size=2560 * 8 // 2,
        head_dim=256,
        vocab_size=262_208,
        sliding_window=1024,
        block_size=8192,
        rope_theta_global=1_000_000,
        rope_theta_local=10_000,    
        rope_scaling_factor_global=8,
        rope_scaling_factor_local=1,
        query_pre_attn_scalar=256,   # ineffective, as it is equal to head_dim
    ),
    "12B": Gemma3Config(
        n_layers=48,
        n_heads=16,
        n_kv_heads=8,
        dim=3840,
        intermediate_size=3840 * 8 // 2,
        head_dim=256,
        vocab_size=262_144,
        sliding_window=1024,
        block_size=131072,
        rope_theta_global=1_000_000,
        rope_theta_local=10_000,
        rope_scaling_factor_global=8,
        rope_scaling_factor_local=1,
        query_pre_attn_scalar=256,
    ),
    "27B": Gemma3Config(
        n_layers=62,
        n_heads=32,
        n_kv_heads=16,
        dim=5376,
        intermediate_size=5376 * 8 // 2,
        head_dim=128,
        vocab_size=262_144,
        sliding_window=1024,
        block_size=131072,
        rope_theta_global=1_000_000,
        rope_theta_local=10_000,
        rope_scaling_factor_global=8,
        rope_scaling_factor_local=1,
        query_pre_attn_scalar=(42 * 128 // 32),
    ),
}
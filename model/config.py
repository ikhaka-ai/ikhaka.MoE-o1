from dataclasses import dataclass, field
#dataclass is used to create custom data types
#this file is essential for the model architecture

@dataclass
class MoEConfig:
    #trunk which is shared across all domain experts
    vocab_size: int = 49_152 #table of 49_152 entries
    d_model: int = 768 #length of vector
    n_layers: int = 12
    n_heads: int = 12 #query heads
    n_kv_heads: int = 4 #less than query heads
    head_dim: int = 64
    seq_len: int = 1024
    rope_theta: float = 10_000.0 #rotary position embeddings
    rms_norm_eps: float = 1e-5

    #experts
    d_ff: int = 2048 #SwiGLU intermediate length per expert
    domains: tuple = ("code", "mathematics", "physics", "general")

    dropout: float = 0.0
    init_std: float = 0.02

    @property
    def num_experts(self) -> int:
        return len(self.domains)

    @property
    def domain_to_id(self) -> dict:
        return {domain: i for i, domain in enumerate(self.domains)}

    def __post_init__(self):
        assert self.n_heads % self.n_kv_heads == 0, ("n_heads must be a scalar multiple of n_kv_heads for GQA grouping")
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"


BASE_CONFIG = MoEConfig() #default setting obj for MoeConfig NN class

DEBUG_CONFIG = MoEConfig(
    vocab_size=1024,
    d_model=128,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    head_dim=2,
    seq_len=64,
    d_ff=256
)

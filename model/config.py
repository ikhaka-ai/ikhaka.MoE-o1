from dataclasses import dataclass, field
#dataclass is used to create custom data types

class MoEConfig:
    #trunk which is shared across all domain experts
    vocab_size: int = 49_152 #table of 49_152 entries
    d_model: int = 768 #length of vector
    n_layers: int = 12
    n_heads: int = 12 #query heads
    n_kv_heads: int = 4 #less than query heads
    head_dim: int = 64
    seq_len: int 1024
    rope_theta: float = 10_000.0 #rotart position embeddings
    rms_norm_eps: float: 1e-5
    

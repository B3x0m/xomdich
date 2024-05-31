from dataclasses import dataclass


@dataclass
class ModelArgs:
    vocab_size: int = 20000
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 512
    ff_dim: int = 512
    norm_eps: float = 1e-8
    batch_size: int = 12
    seq_len: int = 64
    dropout: float = 0.1
    device: str = "cpu"

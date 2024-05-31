import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from typing import Optional, Tuple
from config import ModelArgs


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
#     assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'freqs_cis.shape {freqs_cis.shape} != (x.shape[1], x.shape[-1]) {(x.shape[1], x.shape[-1])}'
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis = freqs_cis.to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
#         super(MultiHeadAttention, self).__init__()
        super().__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert args.hidden_dim % args.n_heads == 0, "d_model must be divisible by num_heads"
        self.n_heads = args.n_heads
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.seq_len = args.seq_len
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_dim // args.n_heads
        
        # Initialize dimensions
        self.d_model = args.hidden_dim # Model's dimension
        self.num_heads = args.n_heads # Number of attention heads
        self.d_k = args.hidden_dim // args.n_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(args.hidden_dim, args.hidden_dim) # Query transformation
        self.W_k = nn.Linear(args.hidden_dim, args.hidden_dim) # Key transformation
        self.W_v = nn.Linear(args.hidden_dim, args.hidden_dim) # Value transformation
        self.W_o = nn.Linear(args.hidden_dim, args.hidden_dim) # Output transformation
        
        self.cache_k = torch.zeros(
            (args.batch_size, args.seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        )
        self.cache_v = torch.zeros(
            (args.batch_size, args.seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        )
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        self.batch_size, self.seq_len, self.hidden_dim = x.size()
        return x.view(self.batch_size, self.seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        self.batch_size, _, self.seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(self.batch_size, self.seq_len, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_dim // args.n_heads

        self.wq = nn.Linear(args.hidden_dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.hidden_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.hidden_dim, bias=False)

        self.cache_k = torch.zeros(
            (args.batch_size, args.seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        )
        self.cache_v = torch.zeros(
            (args.batch_size, args.seq_len, self.n_kv_heads, self.head_dim),
            requires_grad = False
        )
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        start_pos: int = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if start_pos is not None: # if we're performing inference, use kv caching. it'll be 0 to begin with
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
            
            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else: 
            # if we're training, do full sequence length
            keys, values = xk, xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        ff_dim: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = ff_dim * ((hidden_dim + ff_dim - 1) // ff_dim)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = MultiHeadAttention(args)
        self.feed_forward = FeedForward(
                                dim=args.hidden_dim,
                                hidden_dim=4 * args.hidden_dim,
                                ff_dim=args.ff_dim,
                                )
        self.norm1 = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.norm2 = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.hidden_dim
        self.head_dim = args.hidden_dim // args.n_heads
        self.self_attn = MultiHeadAttention(args)
        self.norm1 = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(
            dim=args.hidden_dim,
            hidden_dim=4 * args.hidden_dim,
            ff_dim=args.ff_dim,
        )
        self.ffn_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.dropout_rate = args.dropout
        self.dropout = nn.Dropout(args.dropout)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        start_pos: int = None,
        training = False,
    ):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        x_norm = self.attention_norm(x)
        x_attn = y + self.attention(x_norm, freqs_cis, mask, start_pos)
        h = F.dropout(x_attn, p=self.dropout_rate, training=training)
        
        h_norm = self.ffn_norm(h)
        h_ffwd = self.feed_forward(h_norm)
        out = h + F.dropout(h_ffwd, p=self.dropout_rate, training=training)
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.seq_len = params.seq_len

        self.encoder_embeddings = nn.Embedding(self.vocab_size, params.hidden_dim)
        self.decoder_embeddings = nn.Embedding(self.vocab_size, params.hidden_dim)
        
        self.positional_encoding = PositionalEncoding(params.hidden_dim, params.seq_len)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(self.n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(self.n_layers)])
        
        self.norm = RMSNorm(params.hidden_dim, eps=params.norm_eps)
        self.output = nn.Linear(params.hidden_dim, 
                                params.vocab_size, 
                                bias=False)
        
        self.freqs_cis = precompute_freqs_cis(
            params.hidden_dim // params.n_heads,
            params.seq_len * 2, theta=10000)
        mask = torch.full((params.seq_len, params.seq_len), 
                          float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)
        
#         self.dropout = nn.Dropout(params.dropout)
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, self.seq_len, self.seq_len), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(tgt_mask.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, # specifically for training
                inputs: torch.Tensor, 
                targets: torch.Tensor):
        bsz, seqlen = inputs.shape
        assert inputs.shape == targets.shape
        assert seqlen == self.seq_len
        
        inputs_mask, outputs_mask = self.generate_mask(inputs, targets)
        inputs = self.encoder_embeddings(inputs)
        inputs_emb =self.positional_encoding(inputs) # self.dropout(self.positional_encoding(inputs))
        
        outputs = self.decoder_embeddings(targets)
        freqs_cis = self.freqs_cis
        freqs_cis = self.freqs_cis[:seqlen]
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(inputs_emb, inputs_mask)
            
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(outputs, enc_output, freqs_cis, self.mask, start_pos = None, training = True)
            
        h = self.norm(dec_output)
        
        output = self.output(h).float()
        
        return output

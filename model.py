import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout, max_len):
        super().__init__()
        self.dropout = dropout

        assert embedding_dim % n_heads == 0

        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.n_heads = n_heads
        self.kv_softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.c_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, norm_inputs):
        B, T, C = norm_inputs.shape

        q = self.Q(norm_inputs)  # B, T, C
        k = self.K(norm_inputs)
        v = self.V(norm_inputs)

        # Add extra batch dimension for parallel multi-head processing: (B, T, C) -> (B, n_heads, T, C // n_heads)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Use flash attention implementation
        new_v = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = new_v.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features * 4, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_features * 4, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class GPTBlock(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int, dropout: float, n_heads: int):
        super().__init__()
        assert embedding_dim % n_heads == 0

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
        )
        self.ln1 = LayerNorm(embedding_dim, bias=False)
        self.ln2 = LayerNorm(embedding_dim, bias=False)
        self.mlp = MLP(embedding_dim, embedding_dim, dropout)

    def forward(self, inputs):
        x = inputs + self.multi_head_attention(self.ln1(inputs))
        out = x + self.mlp(self.ln2(x))
        return out


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embedding_dim: int,
        blocks_num: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos = nn.Embedding(max_len, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(embedding_dim, max_len, dropout, n_heads)
                for _ in range(blocks_num)
            ]
        )
        self.proj = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        self.ln_f = LayerNorm(embedding_dim, bias=False)

        # weight-tying
        self.emb.weight = self.proj.weight

    def forward(self, inputs, labels=None):
        _, T = inputs.shape
        embs = self.emb(inputs)
        pos_embs = self.pos(torch.arange(T, device=inputs.device))  # (T,C)
        blocks_output = embs + pos_embs
        for block in self.blocks:
            blocks_output = block(blocks_output)

        blocks_output = self.ln_f(blocks_output)

        logits = self.proj(blocks_output)  # (B,T,vocab_size)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
            return logits, loss
        else:
            return logits, None

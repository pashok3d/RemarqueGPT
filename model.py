import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout, max_len):
        super().__init__()

        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.n_heads = n_heads
        self.kv_softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

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

        attention_weights = (q @ k.transpose(-1, -2)) / math.sqrt(
            q.shape[-1]
        )  # shape: (B, T, T)
        attention_weights = self.attn_dropout(attention_weights)
        attention_weights_masked = attention_weights.masked_fill(
            self.tril[:T, :T] == 0, -torch.inf
        )
        attention_scores = self.kv_softmax(attention_weights_masked)

        new_v = attention_scores @ v  # shape (B, n_heads, T, C // n_heads)

        return new_v.transpose(1, 2).contiguous().view(B, T, C)


class GPTBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, max_len: int, dropout: float = 0.1, n_heads=2
    ):
        super().__init__()
        assert embedding_dim % n_heads == 0

        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
        )

        self.f1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.f_act = nn.SiLU()
        self.f2 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.ff_dropout = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, inputs):
        norm_inputs = self.ln1(inputs)
        attn_out = self.multi_head_attention(norm_inputs)

        # Add residual connection
        x = inputs + attn_out

        norm_x = self.ln2(x)
        ff_out = self.ff_dropout(self.f2(self.f_act(self.f1(norm_x))))
        out = x + ff_out
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
        self.proj = nn.Linear(embedding_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        _, T = inputs.shape
        embs = self.emb(inputs)
        pos_embs = self.pos(torch.arange(T, device=inputs.device))  # (T,C)
        blocks_output = embs + pos_embs
        for block in self.blocks:
            blocks_output = block(blocks_output)
        logits = self.proj(blocks_output)  # (B,T,vocab_size)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
            return logits, loss
        else:
            return logits, None

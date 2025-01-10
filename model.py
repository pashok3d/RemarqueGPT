import torch
from torch import nn
import math


class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim, dropout, max_len):
        super().__init__()

        self.Q = nn.Linear(embedding_dim, head_dim, bias=False)
        self.K = nn.Linear(embedding_dim, head_dim, bias=False)
        self.V = nn.Linear(embedding_dim, head_dim, bias=False)

        self.kv_softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, norm_inputs):

        _, T, _ = norm_inputs.shape

        q = self.Q(norm_inputs)
        k = self.K(norm_inputs)
        v = self.V(norm_inputs)

        attention_weights = (q @ k.transpose(-1, -2)) / math.sqrt(
            q.shape[-1]
        )  # shape: (B, T, T)
        attention_weights_masked = attention_weights.masked_fill(
            self.tril[:T, :T] == 0, -torch.inf
        )
        attention_scores = self.kv_softmax(attention_weights_masked)
        attention_scores = self.attn_dropout(attention_scores)

        return attention_scores @ v


class GPTBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, max_len: int, dropout: float = 0.1, n_heads=2
    ):
        super().__init__()
        head_dim = embedding_dim // n_heads

        # Attention heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(embedding_dim, head_dim, dropout, max_len)
                for _ in range(n_heads)
            ]
        )

        # Feedforward
        self.f1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.f_act = nn.ReLU()
        self.f2 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.ff_dropout = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, inputs):
        norm_inputs = self.ln1(inputs)
        attn_out = torch.concat([h(norm_inputs) for h in self.heads], dim=-1)

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
        embedding_dim: int = 16,
        blocks_num: int = 4,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos = nn.Embedding(max_len, embedding_dim)
        self.blocks = nn.Sequential(
            *[
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
        blocks_output = self.blocks(embs + pos_embs)
        logits = self.proj(blocks_output)  # (B,T,vocab_size)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
            return logits, loss
        else:
            return logits, None

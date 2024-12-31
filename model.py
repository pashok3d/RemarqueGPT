import torch
from torch import nn
import math


class GPTBlock(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int, dropout: float = 0.1):
        # Attention
        super().__init__()
        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.kv_softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

        # Feedforward
        self.f1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.f_act = nn.ReLU()
        self.f2 = nn.Linear(embedding_dim * 4, embedding_dim)
        self.ff_dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, inputs):

        _, T, C = inputs.shape

        norm_inputs = self.ln1(inputs)

        q = self.Q(norm_inputs)
        k = self.K(norm_inputs)
        v = self.V(norm_inputs)

        attention_weights = (q @ k.transpose(-1, -2)) / math.sqrt(C)  # shape: (B, T, T)
        attention_weights_masked = attention_weights.masked_fill(
            self.tril[:T, :T] == 0, -torch.inf
        )
        attention_scores = self.kv_softmax(attention_weights_masked)
        attention_scores = self.attn_dropout(attention_scores)

        new_v = attention_scores @ v + inputs

        x = self.ln2(new_v)
        x = self.ff_dropout(self.f2(self.f_act(self.f1(x)))) + x
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        embedding_dim: int = 16,
        blocks_num: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos = nn.Embedding(max_len, embedding_dim)
        self.blocks = nn.Sequential(
            *[GPTBlock(embedding_dim, max_len) for _ in range(blocks_num)]
        )
        self.proj = nn.Linear(embedding_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        _, T = inputs.shape
        embs = self.emb(inputs)
        pos_embs = self.pos(torch.arange(T, device=device))  # (T,C)
        blocks_output = self.blocks(embs + pos_embs)
        logits = self.proj(blocks_output)  # (B,T,vocab_size)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
            return logits, loss
        else:
            return logits, None

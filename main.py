"""
Building GPT from scratch and training it on all books of Erich Maria Remarque

Available tools: python, pytorch

Tasks:
1. Load data and tokenize to characters
2. Implement GPT model using pytorch
3. Train and evaluate the model

GPT model structure:
1. embedding layer
2. positional encoding
3. blocks
    .1 attention
    .2 feedforward
4. projection
"""

import torch
from torch import nn

# Load and preprocess dataset
with open("dataset/the_dream_room.txt", "r") as f:
    lines = f.readlines()

lines = [l.replace("\xa0", " ").strip() for l in lines if l.strip()]
text = "\n".join(lines)
tokens = sorted(set(text))

id_to_token = {i: token for i, token in enumerate(tokens)}
token_to_id = {token: i for i, token in enumerate(tokens)}


def tokenize(text) -> list[int]:
    return [token_to_id[ch] for ch in text]


def get_dummy_batch():
    return torch.tensor([[1, 2, 3], [4, 5, 6]])


device = "cpu"


class GPTBlock(nn.Module):
    def __init__(self, embedding_dim):
        # Attention
        super().__init__()
        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.kv_softmax = nn.Softmax(dim=-1)

        # Feedforward
        self.f1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.f_act = nn.ReLU()
        self.f2 = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, inputs):
        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)

        attention_weights = q @ k.transpose(-1, -2)  # shape: (B, T, T)
        attention_scores = self.kv_softmax(attention_weights)
        new_v = attention_scores @ v

        x = self.f2(self.f_act(self.f1(new_v)))
        return x


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, max_len: int, embedding_dim: int = 5, blocks_num: int = 2
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos = nn.Embedding(max_len, embedding_dim)
        self.blocks = nn.Sequential(
            *[GPTBlock(embedding_dim) for _ in range(blocks_num)]
        )
        self.proj = nn.Softmax()

    def forward(self, inputs):
        """
        inputs shape is (B, T)
        T = max_len (for simplicty lets assume that all samples in batch are of the same length)
        """
        embs = self.emb(inputs)
        pos_embs = self.pos(inputs)
        blocks_output = self.blocks(embs + pos_embs)
        logits = self.proj(blocks_output)
        return logits


model = GPT(vocab_size=len(tokens), max_len=16)

batch = get_dummy_batch()
batch.to(device)

result = model(batch)

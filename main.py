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
import math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

WINDOW_SIZE = 16
BATCH_SIZE = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

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


def decode(token_ids: list[int]) -> str:
    return "".join([id_to_token[token_id] for token_id in token_ids])


class TextDataset(Dataset):
    def __init__(self, text, context_window_size):
        self.tokens = tokenize(text)

        self.x = []
        self.y = []
        for i in range(len(self.tokens) - context_window_size):
            self.x.append(self.tokens[i : i + context_window_size])
            self.y.append(self.tokens[i + 1 : i + context_window_size + 1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class GPTBlock(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int):
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

        self.register_buffer("tril", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, inputs):

        B, T, C = inputs.shape

        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)

        attention_weights = q @ k.transpose(-1, -2)  # shape: (B, T, T)
        attention_weights_masked = attention_weights.masked_fill(
            self.tril[:T, :T] == 0, -torch.inf
        )
        attention_scores = self.kv_softmax(attention_weights_masked)
        new_v = attention_scores @ v

        x = self.f2(self.f_act(self.f1(new_v)))
        return x


class GPT(nn.Module):
    def __init__(
        self, vocab_size: int, max_len: int, embedding_dim: int = 5, blocks_num: int = 2
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
        B, T = inputs.shape
        embs = self.emb(inputs)
        pos_embs = self.pos(torch.arange(T, device=device))  # (T,C)
        blocks_output = self.blocks(embs + pos_embs)
        logits = self.proj(blocks_output)  # (B,T,vocab_size)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
            return logits, loss
        else:
            return logits, None


ds = TextDataset(text, WINDOW_SIZE)
dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

model = GPT(vocab_size=len(tokens), max_len=WINDOW_SIZE)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters())


model.train()
epoch_loss = 0
steps_n = 0
with torch.no_grad():
    for batch in tqdm(dataloader):
        input, labels = batch[0].to(device), batch[1].to(device)
        output, loss = model(input, labels)
        epoch_loss += loss.item()
        steps_n += 1
    avg_loss = epoch_loss / steps_n
expected_init_loss = -math.log(1 / 74)
print(f"initial train loss: {avg_loss:.3f}, with expected of {expected_init_loss:.3f}")

for epoch in range(5):
    model.train()
    epoch_loss = 0
    steps_n = 0
    for batch in tqdm(dataloader):
        input, labels = batch[0].to(device), batch[1].to(device)
        output, loss = model(input, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        steps_n += 1
    avg_loss = epoch_loss / steps_n
    print(f"epoch {epoch} train loss: {avg_loss:.3f}")

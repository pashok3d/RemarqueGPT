"""
Building GPT from scratch and training it on all books of Erich Maria Remarque

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

import tiktoken
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_datasets, generate_text
from tokenizer import Tokenizer
from model import GPT
from torch.utils.data import ConcatDataset

WINDOW_SIZE = 5
BATCH_SIZE = 2
EPOCHS = 10
LR = 0.005

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "learning_rate": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "window_size": WINDOW_SIZE,
}

# Prepare tokenizer
enc = tiktoken.get_encoding("o200k_base")
tokenizer = Tokenizer(enc)

dataset_lines = []
with open("dataset/The_Dream_Room_1920_AST_978-5-17-071518-3.txt", "r") as f:
    dataset_lines.extend(f.readlines())
text = "\n".join(dataset_lines)

tokenizer.fit(text)

tokenizer.dump("tokenizer.txt")

# Prepare datasets
ds_list = get_datasets(
    [
        "dataset/The_Dream_Room_1920_AST_978-5-17-071518-3-train.txt",
    ],
    WINDOW_SIZE,
    tokenizer,
)
train_ds = ConcatDataset(ds_list)
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

dev_ds_list = get_datasets(
    [
        "dataset/The_Dream_Room_1920_AST_978-5-17-071518-3-dev.txt",
    ],
    WINDOW_SIZE,
    tokenizer,
)
dev_ds = ConcatDataset(dev_ds_list)
dev_dataloader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

model = GPT(
    vocab_size=tokenizer.vocab_size, max_len=WINDOW_SIZE, blocks_num=2, embedding_dim=4
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

model.train()
epoch_loss = 0
steps_n = 0
with torch.no_grad():
    for batch in tqdm(train_dataloader):
        input, labels = batch[0].to(device), batch[1].to(device)
        output, loss = model(input, labels)
        epoch_loss += loss.item()
        steps_n += 1
    avg_loss = epoch_loss / steps_n
expected_init_loss = -math.log(1 / 74)
print(f"initial train loss: {avg_loss:.3f}, with expected of {expected_init_loss:.3f}")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    val_epoch_loss = 0
    steps_n = 0
    val_steps_n = 0

    for batch in tqdm(train_dataloader):
        input, labels = batch[0].to(device), batch[1].to(device)
        output, loss = model(input, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        steps_n += 1

    avg_loss = epoch_loss / steps_n
    print(f"epoch {epoch} train loss: {avg_loss:.3f}")

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            input, labels = batch[0].to(device), batch[1].to(device)
            output, loss = model(input, labels)
            val_epoch_loss += loss.item()
            val_steps_n += 1

    avg_val_loss = val_epoch_loss / val_steps_n
    print(f"epoch {epoch} val loss: {avg_val_loss:.3f}")

torch.save(model.state_dict(), "model/gpt.pt")

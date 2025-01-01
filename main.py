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

import wandb
import torch
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import TextDataset
from model import GPT
from torch.utils.data import ConcatDataset

WINDOW_SIZE = 32
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.005

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "learning_rate": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "window_size": WINDOW_SIZE,
}

run = wandb.init(project="remark-gpt", config=config)

# Prepare tokenizer
dataset_lines = []
with open("dataset/The_Dream_Room_1920_AST_978-5-17-071518-3.txt", "r") as f:
    dataset_lines.extend(f.readlines())
with open("dataset/Station_at_the_Horizon_1928_AST_978-5-17-133322-5.txt", "r") as f:
    dataset_lines.extend(f.readlines())
text = "\n".join(dataset_lines)
tokens = sorted(set(text))
id_to_token = {i: token for i, token in enumerate(tokens)}
token_to_id = {token: i for i, token in enumerate(tokens)}

train_ds1 = TextDataset(
    "dataset/The_Dream_Room_1920_AST_978-5-17-071518-3-train.txt",
    WINDOW_SIZE,
    token_to_id,
)
train_ds2 = TextDataset(
    "dataset/Station_at_the_Horizon_1928_AST_978-5-17-133322-5.txt",
    WINDOW_SIZE,
    token_to_id,
)
train_ds = ConcatDataset([train_ds1, train_ds2])
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

dev_ds1 = TextDataset(
    "dataset/The_Dream_Room_1920_AST_978-5-17-071518-3-dev.txt",
    WINDOW_SIZE,
    token_to_id,
)
dev_ds2 = TextDataset(
    "dataset/Station_at_the_Horizon_1928_AST_978-5-17-133322-5-dev.txt",
    WINDOW_SIZE,
    token_to_id,
)
dev_ds = ConcatDataset([dev_ds1, dev_ds2])
dev_dataloader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

test_ds1 = TextDataset(
    "dataset/The_Dream_Room_1920_AST_978-5-17-071518-3-test.txt",
    WINDOW_SIZE,
    token_to_id,
)
test_ds2 = TextDataset(
    "dataset/Station_at_the_Horizon_1928_AST_978-5-17-133322-5-test.txt",
    WINDOW_SIZE,
    token_to_id,
)
test_ds = ConcatDataset([test_ds1, test_ds2])
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = GPT(vocab_size=len(tokens), max_len=WINDOW_SIZE, blocks_num=2)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

wandb.watch(model, log_freq=5000)

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
    test_epoch_loss = 0
    test_steps_n = 0

    for batch in tqdm(train_dataloader):
        input, labels = batch[0].to(device), batch[1].to(device)
        output, loss = model(input, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        steps_n += 1
        run.log({"train_loss": loss.item()})

    avg_loss = epoch_loss / steps_n
    print(f"epoch {epoch} train loss: {avg_loss:.3f}")

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            input, labels = batch[0].to(device), batch[1].to(device)
            output, loss = model(input, labels)
            val_epoch_loss += loss.item()
            val_steps_n += 1

        for batch in tqdm(test_dataloader):
            input, labels = batch[0].to(device), batch[1].to(device)
            output, loss = model(input, labels)
            test_epoch_loss += loss.item()
            test_steps_n += 1

    avg_val_loss = val_epoch_loss / val_steps_n
    avg_test_loss = test_epoch_loss / test_steps_n
    print(f"epoch {epoch} val loss: {avg_val_loss:.3f}")
    print(f"epoch {epoch} test loss: {avg_test_loss:.3f}")
    run.log({"epoch_train_loss": avg_loss, "epoch_val_loss": avg_val_loss})

torch.save(model.state_dict(), "model/gpt.pt")

artifact = wandb.Artifact("model", type="model")
artifact.add_file("model/gpt.pt")
run.log_artifact(artifact)

wandb.finish()

"""
Building GPT from scratch and training it on all books of Erich Maria Remarque
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from utils import get_datasets, generate_text
from model import GPT
from torch.utils.data import ConcatDataset
from transformers import get_linear_schedule_with_warmup
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
    processors,
    normalizers,
)
from torch.nn.utils import clip_grad_norm_

LOG_WANDB = False
WINDOW_SIZE = 256
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-4
EMBEDDING_DIM = 128
BLOCKS_NUM = 4
HEADS_NUM = 4
DROPOUT = 0.2
VOCAB_SIZE = 2024
MAX_GRAD_NORM = 1.0
VALIDATION_INTERVAL = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "learning_rate": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "window_size": WINDOW_SIZE,
    "embedding_dim": EMBEDDING_DIM,
    "blocks_num": BLOCKS_NUM,
    "heads_num": HEADS_NUM,
    "dropout": DROPOUT,
    "vocab_size": VOCAB_SIZE,
    "max_grad_norm": MAX_GRAD_NORM,
    "device": device,
}

if LOG_WANDB:
    import wandb

    run = wandb.init(project="remark-gpt", config=config)

# Prepare tokenizer
dataset_lines = []

dataset_paths = [
    "dataset/All_Quiet_on_the_Western_Front_1929_AST_978-5-17-105639-1.txt",
    "dataset/All_Quiet_on_the_Western_Front_1929_AST_978-5-17-137374-0.txt",
    "dataset/Station_at_the_Horizon_1928_AST_978-5-17-133322-5.txt",
    "dataset/The_Dream_Room_1920_AST_978-5-17-071518-3.txt",
    "dataset/The_Road_Back_1931.txt",
    "dataset/Tree_Comrades_1936_978-5-17-004252-4.txt",
    "dataset/Tree_Comrades_1936_978-5-17-056963-2.txt",
    "dataset/Tree_Comrades_1936_978-5-17-108545-2.txt",
]

for path in [ds_path.replace(".txt", "-train.txt") for ds_path in dataset_paths]:
    with open(path, "r") as f:
        dataset_lines.extend(f.readlines())

alphabet = list(set("".join(dataset_lines)))

tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE)
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.train_from_iterator(dataset_lines, trainer=trainer)

# Prepare dataloaders
ds_list = get_datasets(
    [ds_path.replace(".txt", "-train.txt") for ds_path in dataset_paths],
    WINDOW_SIZE,
    tokenizer,
)

train_ds = ConcatDataset(ds_list)
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

dev_dataset_paths = [ds_path.replace(".txt", "-dev.txt") for ds_path in dataset_paths]

dev_ds_list = get_datasets(
    dev_dataset_paths,
    WINDOW_SIZE,
    tokenizer,
)

names_with_dev_dataloaders = []

for i, dev_ds in enumerate(dev_ds_list):
    dev_dataloader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)
    names_with_dev_dataloaders.append(
        (
            dev_dataset_paths[i].replace("dataset/", "").replace(".txt", ""),
            dev_dataloader,
        )
    )

model = GPT(
    vocab_size=tokenizer.vocab_size, max_len=WINDOW_SIZE, blocks_num=2, embedding_dim=4
)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Scheduler with warmup and linear decay
TOTAL_STEPS = EPOCHS * len(train_dataloader)
WARMUP_STEPS = int(0.05 * TOTAL_STEPS)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS
)

if LOG_WANDB:
    wandb.watch(model, log_freq=2500, log="all")

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
expected_init_loss = -math.log(1 / tokenizer.get_vocab_size())
print(f"initial train loss: {avg_loss:.3f}, with expected of {expected_init_loss:.3f}")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    total_steps = 0

    for batch in tqdm(train_dataloader):
        input, labels = batch[0].to(device), batch[1].to(device)
        output, loss = model(input, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        total_steps += 1
        if LOG_WANDB:
            run.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        if total_steps % VALIDATION_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for ds_name, dev_dataloader in names_with_dev_dataloaders:
                    val_steps_n = 0
                    val_epoch_loss = 0
                    for batch in dev_dataloader:
                        input, labels = batch[0].to(device), batch[1].to(device)
                        output, loss = model(input, labels)
                        val_epoch_loss += loss.item()
                        val_steps_n += 1
                    avg_val_loss = val_epoch_loss / val_steps_n
                    print(f"val loss for {ds_name}: {avg_val_loss:.3f}")
                    metric_name = f"avg_val_loss_{ds_name}"
                    if LOG_WANDB:
                        run.log({metric_name: avg_val_loss}, commit=False)
            model.train()

torch.save(model.state_dict(), "model/gpt.pt")

if LOG_WANDB:
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("model/gpt.pt")
    run.log_artifact(artifact)
    run.finish()

model.eval()
prompt = "Привет, любовь моя"
generated_text = generate_text(
    model, tokenizer, prompt, device, WINDOW_SIZE, max_tokens=250, temperature=0.7
)

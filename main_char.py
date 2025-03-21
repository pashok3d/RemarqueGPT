"""
Building GPT from scratch and training it on all books of Erich Maria Remarque
"""

import collections
import math

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import get_inverse_sqrt_schedule

from char_tokenizer import CharTokenizer
from model import GPT
from utils import generate_text, get_datasets

LOG_WANDB = True
WINDOW_SIZE = 512
BATCH_SIZE = 256
EPOCHS = 5
LR = 5e-4
EMBEDDING_DIM = 128
BLOCKS_NUM = 12
HEADS_NUM = 4
DROPOUT = 0.15
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.05
USE_BFLOAT16 = False

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
    "max_grad_norm": MAX_GRAD_NORM,
    "warmup_fraction": WARMUP_FRACTION,
    "device": device,
    "use_bfloat16": USE_BFLOAT16,
}

if LOG_WANDB:
    import wandb

    run = wandb.init(project="remark-gpt-char", config=config)

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
    "dataset/A_Time_to_Love_and_a_Time_to_Die.txt",
    "dataset/Arch_of_Triumph_1945.txt",
    "dataset/Flotsam_1939.txt",
    "dataset/Heaven_Has_No_Favorites.txt",
    "dataset/Shadows_in_Paradise.txt",
    "dataset/Spark_of_Life_1952.txt",
    "dataset/The_Black_Obelisk.txt",
    "dataset/The_Promised_Land_Vagrius_978-5-9697-0386-5.txt",
    "dataset/The_Night_in_Lisbon.txt",
    "dataset/Gam.txt",
]

for path in dataset_paths:
    with open(path, "r") as f:
        dataset_lines.extend(f.readlines())

text = "\n".join(dataset_lines)
c = collections.Counter(text.lower())
del text
alphabet = [char for char, _ in c.most_common()]

tokenizer = CharTokenizer(alphabet)

# Save tokenizer
tokenizer.save("tokenizer")

# Prepare dataloaders
ds_list = get_datasets(
    [ds_path.replace(".txt", "-train.txt") for ds_path in dataset_paths],
    WINDOW_SIZE,
    tokenizer,
    device,
)

train_ds = ConcatDataset(ds_list)
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

dev_dataset_paths = [ds_path.replace(".txt", "-dev.txt") for ds_path in dataset_paths]

dev_ds_list = get_datasets(dev_dataset_paths, WINDOW_SIZE, tokenizer, device)

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
    vocab_size=tokenizer.get_vocab_size(),
    max_len=WINDOW_SIZE,
    embedding_dim=EMBEDDING_DIM,
    blocks_num=BLOCKS_NUM,
    n_heads=HEADS_NUM,
    dropout=DROPOUT,
)
model.to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)

# Scheduler with warmup and linear decay
total_steps = EPOCHS * len(train_dataloader)
warmup_steps = int(WARMUP_FRACTION * total_steps)
val_interval = min(500, len(train_dataloader) * 0.25)

scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=warmup_steps)

model.train()

epoch_loss = 0
steps_n = 0
with torch.no_grad():
    for batch in tqdm(names_with_dev_dataloaders[0][1]):
        input, labels = batch[0].to(device), batch[1].to(device)
        if USE_BFLOAT16:
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                output, loss = model(input, labels)
        else:
            output, loss = model(input, labels)
        epoch_loss += loss.item()
        steps_n += 1
    avg_loss = epoch_loss / steps_n
expected_init_loss = -math.log(1 / tokenizer.get_vocab_size())
print(f"initial train loss: {avg_loss:.3f}, with expected of {expected_init_loss:.3f}")

total_steps = 0
for epoch in range(EPOCHS):
    model.train()
    for batch in tqdm(train_dataloader):
        input, labels = batch[0].to(device), batch[1].to(device)
        if USE_BFLOAT16:
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                output, loss = model(input, labels)
        else:
            output, loss = model(input, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if LOG_WANDB:
            run.log({"train_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

        if total_steps % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for ds_name, dev_dataloader in names_with_dev_dataloaders:
                    val_steps_n = 0
                    val_epoch_loss = 0
                    for batch in dev_dataloader:
                        input, labels = batch[0].to(device), batch[1].to(device)
                        if USE_BFLOAT16:
                            with torch.amp.autocast(
                                device_type=device, dtype=torch.bfloat16
                            ):
                                output, loss = model(input, labels)
                        else:
                            output, loss = model(input, labels)
                        val_epoch_loss += loss.item()
                        val_steps_n += 1
                    avg_val_loss = val_epoch_loss / val_steps_n
                    print(f"val loss for {ds_name}: {avg_val_loss:.3f}")
                    metric_name = f"avg_val_loss_{ds_name}"
                    if LOG_WANDB:
                        run.log({metric_name: avg_val_loss}, commit=False)

                prompt = "она смотрела на него сквозь сигаретный дым, и в ее глазах"
                generated_text = generate_text(
                    model, tokenizer, prompt, device, WINDOW_SIZE, max_tokens=250
                )
                print(f"generated text: {generated_text}")

            torch.save(model.state_dict(), "gpt.pt")
            model.train()

        total_steps += 1

torch.save(model.state_dict(), "gpt.pt")

if LOG_WANDB:
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file("gpt.pt")
    artifact.add_file("tokenizer")
    run.log_artifact(artifact)
    run.finish()

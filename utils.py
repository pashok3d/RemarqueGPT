import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer


def generate_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    window_size: int,
    max_tokens: int = 10,
    temperature: float = 1.0,
) -> str:
    """Generate text using the trained GPT model."""
    model.eval()
    context = tokenizer.encode(prompt)
    generated = list(context)

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor(context[-window_size:]).unsqueeze(0).to(device)
            logits, _ = model(x)
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            context = generated

    return tokenizer.decode(generated)


class TextDataset(Dataset):
    def __init__(self, path, context_window_size, tokenizer: Tokenizer):
        # Load dataset
        with open(path, "r") as f:
            lines = f.readlines()

        text = "\n".join(lines)

        self.tokens = tokenizer.encode(text)

        self.x = []
        self.y = []
        for i in range(len(self.tokens) - context_window_size):
            self.x.append(self.tokens[i : i + context_window_size])
            self.y.append(self.tokens[i + 1 : i + context_window_size + 1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


def get_datasets(paths, context_window_size, tokenizer: Tokenizer):
    datasets = []

    for path in paths:
        dataset = TextDataset(path, context_window_size, tokenizer)
        datasets.append(dataset)

    return datasets

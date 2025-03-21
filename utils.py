from typing import Union

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from char_tokenizer import CharTokenizer


def generate_text(
    model,
    tokenizer: CharTokenizer,
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


class CharDataset(Dataset):
    def __init__(self, path, context_window_size, tokenizer: CharTokenizer, device):
        """
        Args:
            path (str): Path to the text file
            context_window_size (int): Size of the context window
            tokenizer: The tokenizer to use for encoding the text
        """
        self.path = path
        self.context_window_size = context_window_size
        self.tokenizer = tokenizer
        self.device = device

        # Store file size for length calculation
        with open(path, "r") as f:
            self.text = f.read()

        self.total_tokens = len(self.text)

    def __len__(self):
        return self.total_tokens - self.context_window_size

    def __getitem__(self, idx):
        """
        Get input and target sequences by tokenizing text on-the-fly.

        Args:
            idx (int): Index to start the sequence

        Returns:
            tuple: (input_sequence, target_sequence)
        """
        x = self.text[idx : idx + self.context_window_size]
        y = self.text[idx + 1 : idx + self.context_window_size + 1]

        x = self.tokenizer.encode(x)
        y = self.tokenizer.encode(y)

        x, y = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        if self.device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


class TextDataset(Dataset):
    def __init__(self, path, context_window_size, tokenizer: Tokenizer, device):
        """
        Args:
            path (str): Path to the text file
            context_window_size (int): Size of the context window
            tokenizer: The tokenizer to use for encoding the text
        """
        self.path = path
        self.context_window_size = context_window_size
        self.tokenizer = tokenizer
        self.device = device

        # Store file size for length calculation
        with open(path, "r") as f:
            self.text = f.read()

        self.tokens = self.tokenizer.encode(self.text).ids
        self.total_tokens = len(self.tokens)

    def __len__(self):
        return self.total_tokens - self.context_window_size

    def __getitem__(self, idx):
        """
        Get input and target sequences by tokenizing text on-the-fly.

        Args:
            idx (int): Index to start the sequence

        Returns:
            tuple: (input_sequence, target_sequence)
        """
        x = self.tokens[idx : idx + self.context_window_size]
        y = self.tokens[idx + 1 : idx + self.context_window_size + 1]

        x, y = torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

        if self.device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


def get_datasets(
    paths, context_window_size, tokenizer: Union[CharTokenizer, Tokenizer], device
) -> Union[list[CharDataset], list[TextDataset]]:
    datasets = []

    if isinstance(tokenizer, CharTokenizer):
        for path in paths:
            dataset = CharDataset(path, context_window_size, tokenizer, device)
            datasets.append(dataset)
    else:
        for path in paths:
            dataset = TextDataset(path, context_window_size, tokenizer, device)
            datasets.append(dataset)

    return datasets

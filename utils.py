import torch


def tokenize(text, token_to_id) -> list[int]:
    return [token_to_id[ch] for ch in text]


def decode(token_ids: list[int], id_to_token) -> str:
    return "".join([id_to_token[token_id] for token_id in token_ids])


def generate_text(
    model,
    prompt: str,
    device: str,
    window_size: int,
    max_tokens: int = 10,
    temperature: float = 1.0,
) -> str:
    """Generate text using the trained GPT model."""
    model.eval()
    context = tokenize(prompt)
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

    return decode(context)

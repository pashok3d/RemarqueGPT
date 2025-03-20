import json


class Encoding:
    def __init__(self, ids, type_ids, attention_mask):
        self.ids = ids
        self.type_ids = type_ids
        self.attention_mask = attention_mask


class CharTokenizer:
    def __init__(self, alphabet):
        self.char2idx = {}
        self.idx2char = {}
        for idx, char in enumerate(alphabet):
            assert len(char) == 1
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def get_vocab_size(self):
        return len(self.char2idx)

    def encode(self, text) -> Encoding:
        return Encoding(
            ids=[self.char2idx[char] for char in text.lower()],
            type_ids=[0] * len(text),
            attention_mask=[1] * len(text),
        )

    def decode(self, tokens) -> str:
        return "".join([self.idx2char[idx] for idx in tokens])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"char2idx": self.char2idx, "idx2char": self.idx2char}, f)

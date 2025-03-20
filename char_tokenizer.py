import json


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

    def encode(self, text) -> list[int]:
        return [self.char2idx[char] for char in text.lower()]

    def decode(self, tokens) -> str:
        return "".join([self.idx2char[idx] for idx in tokens])

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"char2idx": self.char2idx, "idx2char": self.idx2char}, f)

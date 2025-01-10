class Tokenizer:
    def __init__(self, encoding):
        self.encoding = encoding
        self._is_fit = False

    @property
    def vocab_size(self):
        if not self._is_fit:
            raise Exception("Tokenizer not fitted yet. Call fit() method first.")

        return len(self.id_to_token)

    def dump(self, path: str):
        if not self._is_fit:
            raise Exception("Tokenizer not fitted yet. Call fit() method first.")

        with open(path, "w") as f:
            for id, tik_token in self.id_to_tik_token.items():
                f.write(f"{id}\t{tik_token}\t{self.encoding.decode([tik_token])}\n")

    def fit(self, text: str) -> None:
        tik_tokens = self.encoding.encode_ordinary(text)
        tik_tokens = set(tik_tokens)

        # Map local tokens to tiktoken tokens
        self.id_to_tik_token = {
            i: tiktok_token for i, tiktok_token in enumerate(tik_tokens)
        }
        # Map tiktoken tokens to local tokens
        self.tik_token_to_id = {
            tiktok_token: i for i, tiktok_token in enumerate(tik_tokens)
        }

        self._is_fit = True

    def encode(self, text: str):
        if not self._is_fit:
            raise Exception("Tokenizer not fitted yet. Call fit() method first.")

        # Encode text
        tik_tokens = self.encoding.encode_ordinary(text)

        # Map tiktoken tokens to local tokens
        token_ids = [self.tik_token_to_id[token] for token in tik_tokens]

        return token_ids

    def decode(self, token_ids: list[int]):
        if not self._is_fit:
            raise Exception("Tokenizer not fitted yet. Call fit() method first.")

        # Map local tokens to tiktoken tokens
        tik_tokens = [self.id_to_tik_token[token_id] for token_id in token_ids]

        # Decode tokens
        return self.encoding.decode(tik_tokens)

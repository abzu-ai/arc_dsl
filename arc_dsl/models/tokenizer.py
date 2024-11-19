import json
import re

from ..interpreter import DSL, DEFAULT_ENV, canonicalize


PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
COMPLETION_TOKEN = "<COMPLETION>"
END_TOKEN = "<END>"
PARAM_TOKEN = " ?"
MANY_PARAM_TOKEN = " ¿"
SPECIAL_TOKEN_6 = "<UNUSED-SPECIAL-6>"
SPECIAL_TOKEN_7 = "<UNUSED-SPECIAL-7>"
SPECIAL_TOKEN_8 = "<UNUSED-SPECIAL-8>"


def _build_vocab():
    special_tokens = [
        PAD_TOKEN,
        START_TOKEN,
        COMPLETION_TOKEN,
        END_TOKEN,
        PARAM_TOKEN,
        MANY_PARAM_TOKEN,
        SPECIAL_TOKEN_6,
        SPECIAL_TOKEN_7,
        SPECIAL_TOKEN_8,
    ]

    # Build vocabulary from special tokens, functions in env, and numbers
    vocab = []

    # Add special tokens
    vocab.extend(special_tokens)

    # Symbols defined in environment
    vocab.extend([f" {name}" for name in DEFAULT_ENV.keys()])

    # Special forms from the interpreter
    vocab.extend([" (", ")", " SETPARAM", " LAMBDA", " COND", "'", " QUOTE", " EVAL"])

    # Add booleans and numbers 0-30
    vocab.extend(["TRUE", "FALSE"])
    vocab.extend([f" {i}" for i in range(31)])

    return vocab


class GridlispTokenizer:
    lparen_pattern = re.compile(r"\s*\(\s*")

    def __init__(self, ids_to_tokens: dict[int, str] = None):
        """
        Initializes the GridlispTokenizer.

        Args:
            ids_to_tokens (dict[int, str], optional): A dictionary mapping token IDs to tokens.
                If None, the vocabulary is generated from the DSL.
        """
        if ids_to_tokens is None:
            vocab = _build_vocab()
            self.id_to_token = {i: token for i, token in enumerate(vocab)}
        else:
            vocab = list(ids_to_tokens.values())
            self.id_to_token = ids_to_tokens

        self.token_to_id = {token: id for id, token in self.id_to_token.items()}

        # Sort tokens by length (longest first) to ensure proper tokenization
        token_patterns = sorted(
            [re.escape(token) for token in vocab], key=len, reverse=True
        )
        self.pattern = re.compile("|".join(token_patterns))

        self.start_token_id = self.token_to_id[START_TOKEN]
        self.end_token_id = self.token_to_id[END_TOKEN]
        self.completion_token_id = self.token_to_id[COMPLETION_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, sentence: str) -> list[int]:
        sentence = self.make_tokenizable(sentence)
        token_ids = []

        # Find all tokens using regex pattern
        for token in self.pattern.findall(sentence):
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                raise ValueError(f"Unknown token: {token}")

        # Verify full string was tokenized
        unmatched_tokens = re.sub(self.pattern, "", sentence).strip()
        if unmatched_tokens:
            raise ValueError(f"Unknown tokens: {unmatched_tokens}")

        return token_ids

    def decode(self, encoding: list[int]) -> str:
        try:
            return "".join(self.id_to_token[id] for id in encoding)
        except KeyError as e:
            raise ValueError(f"Unknown token ID: {e.args[0]}")

    def make_tokenizable(self, sentence: str) -> str:
        sentence = canonicalize(
            sentence, preserve_these=[PARAM_TOKEN, MANY_PARAM_TOKEN]
        )
        sentence = sentence.replace("\n", " ")
        sentence = self.lparen_pattern.sub(" ( ", sentence)

        if not sentence.startswith(" "):
            sentence = " " + sentence

        return sentence

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.id_to_token, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            id_to_token = json.load(f)

        # Convert string keys back to integers
        id_to_token = {int(k): v for k, v in id_to_token.items()}
        return cls(ids_to_tokens=id_to_token)

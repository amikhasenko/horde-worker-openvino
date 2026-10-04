"""openai CLIP BPE tokenizer, vendored so the runtime never imports the ``clip``
package (whose ``__init__`` pulls in torch).

Copied verbatim from ``AI-Horde-Worker/worker/ov_alchemy/clip_tokenizer.py`` so this
worker stays independent of the Alchemist backend.  Keep the two in sync.

Verbatim from ``clip/simple_tokenizer.py`` (MIT, OpenAI) plus :func:`tokenize`, which
reproduces ``clip.tokenize(..., truncate=True)`` exactly:

* ``context_length`` = 77, ``sot``/``eot`` wrap the sequence, right-padded with 0;
* over-long sequences are truncated and the last token forced to eot.

Unlike the torch original this returns a NumPy ``int64`` array instead of a tensor.
"""
from __future__ import annotations

import gzip
import html
import os
from functools import lru_cache

import ftfy
import numpy as np
import regex as re


@lru_cache()
def default_bpe() -> str:
    # Prefer the copy the converter wrote next to the IR.
    root = os.environ.get("HORDE_OV_DIR")
    if root:
        candidate = os.path.join(root, "bpe_simple_vocab_16e6.txt.gz")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """Reversible byte<->unicode table from the original GPT-2 BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    return re.sub(r"\s+", " ", text).strip()


class SimpleTokenizer:
    def __init__(self, bpe_path: str = None):
        bpe_path = bpe_path or default_bpe()
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with gzip.open(bpe_path) as f:
            merges = f.read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        return (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )


_tokenizer: SimpleTokenizer | None = None


def get_tokenizer(bpe_path: str = None) -> SimpleTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = SimpleTokenizer(bpe_path)
    return _tokenizer


def tokenize(texts, context_length: int = 77, truncate: bool = True, bpe_path: str = None) -> np.ndarray:
    """Reproduce ``clip.tokenize(texts, truncate=True)`` as an int64 NumPy array."""
    if isinstance(texts, str):
        texts = [texts]
    tokenizer = get_tokenizer(bpe_path)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, : len(tokens)] = tokens
    return result

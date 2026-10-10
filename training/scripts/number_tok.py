"""How OLMo3 / DeepSeek-V4-Flash / Kimi-K2.6 tokenize numbers.

For math/proof work the digit-grouping rule matters: single-digit tokenization
keeps arithmetic regular but is token-hungry; greedy multi-digit grouping is
compact but makes place-value inconsistent (e.g. 1000 -> 100|0). We tokenize a
fixed battery of number strings with each tokenizer and print the pieces.
"""

from __future__ import annotations

import base64

import tiktoken
from transformers import AutoTokenizer

OLMO = "/home/4/uq06834/Working/hdd/BigModels/Olmo-3-1025-7B"
DEEPSEEK = "/home/4/uq06834/Working/hdd/BigModels/DeepSeek-V4-Flash"
KIMI_TIKTOKEN = "/home/4/uq06834/Working/hdd/BigModels/Kimi-K2.6-tok/tiktoken.model"

# Kimi pretokenization regex (from tokenization_kimi.py); note \p{N}{1,3}.
KIMI_PAT = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

TESTS = [
    "0", "7", "10", "42", "100", "123", "999", "1000", "1024",
    "12345", "100000", "1000000", "31415926",
    "3.14159", "2.71828", "1/3", "x^2", "2^10", "1+2=3",
    "The answer is 1000000.", "n=998244353", "mod 10^9+7",
]


def load_kimi() -> tiktoken.Encoding:
    ranks: dict[bytes, int] = {}
    with open(KIMI_TIKTOKEN) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64, rank = line.split()
            ranks[base64.b64decode(b64)] = int(rank)
    return tiktoken.Encoding(name="kimi", pat_str=KIMI_PAT,
                             mergeable_ranks=ranks, special_tokens={})


def hf_pieces(tok, s: str) -> list[str]:
    ids = tok.encode(s, add_special_tokens=False)
    return [tok.decode([i]) for i in ids]


def kimi_pieces(enc: tiktoken.Encoding, s: str) -> list[str]:
    return [enc.decode([i]) for i in enc.encode(s)]


def main() -> None:
    olmo = AutoTokenizer.from_pretrained(OLMO, trust_remote_code=True)
    deepseek = AutoTokenizer.from_pretrained(DEEPSEEK, trust_remote_code=True)
    kimi = load_kimi()

    def fmt(pieces: list[str]) -> str:
        return f"[{len(pieces):>2}] " + "|".join(repr(p)[1:-1] for p in pieces)

    print(f"{'input':<24} {'tokenizer':<10} pieces")
    print("-" * 90)
    for s in TESTS:
        print(f"{s!r:<24} {'OLMo3':<10} {fmt(hf_pieces(olmo, s))}")
        print(f"{'':<24} {'DeepSeek':<10} {fmt(hf_pieces(deepseek, s))}")
        print(f"{'':<24} {'Kimi':<10} {fmt(kimi_pieces(kimi, s))}")
        print()


if __name__ == "__main__":
    main()

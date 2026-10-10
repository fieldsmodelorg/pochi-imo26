"""Byte-level vocab overlap between OLMo 3 and candidate donor tokenizers.

Different tokenizers serialize their vocab strings differently (HF byte-level
BPE uses the GPT-2 bytes->unicode map; tiktoken stores raw bytes as base64).
A fair "shared token" count must compare the *raw byte sequences* of each
token, not the surface strings. This script normalizes every vocab to a set of
``bytes`` and reports pairwise overlap.
"""

from __future__ import annotations

import base64
import functools

from transformers import AutoTokenizer

OLMO = "/home/4/uq06834/Working/hdd/BigModels/Olmo-3-1025-7B"
DEEPSEEK = "/home/4/uq06834/Working/hdd/BigModels/DeepSeek-V4-Flash"
KIMI_TIKTOKEN = "/home/4/uq06834/Working/hdd/BigModels/Kimi-K2.6-tok/tiktoken.model"


@functools.lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2 bytes_to_unicode: printable-unicode-char -> raw byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def hf_vocab_bytes(path: str) -> set[bytes]:
    """Raw byte sequences for every *non-special* token in an HF byte-level BPE vocab."""
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    special = set(tok.all_special_tokens)
    dec = _byte_decoder()
    out: set[bytes] = set()
    skipped = 0
    for s in tok.get_vocab():
        if s in special:
            continue
        try:
            out.add(bytes(dec[ch] for ch in s))
        except KeyError:
            skipped += 1  # token with chars outside the byte-level alphabet
    print(f"  {path.split('/')[-1]}: {len(out)} byte-tokens ({skipped} non-byte-level skipped)")
    return out


def tiktoken_vocab_bytes(path: str) -> set[bytes]:
    out: set[bytes] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64 = line.split()[0]
            out.add(base64.b64decode(b64))
    print(f"  {path.split('/')[-2]}: {len(out)} byte-tokens (tiktoken)")
    return out


def report(name: str, base: set[bytes], donor: set[bytes]) -> None:
    shared = base & donor
    print(f"\n=== OLMo3 vs {name} ===")
    print(f"  base(OLMo3) vocab : {len(base)}")
    print(f"  donor({name}) vocab: {len(donor)}")
    print(f"  shared (byte-exact): {len(shared)}")
    print(f"  shared / base  = {len(shared) / len(base):.3%}")
    print(f"  shared / donor = {len(shared) / len(donor):.3%}")
    print(f"  new tokens in donor (need OMP) = {len(donor) - len(shared)} "
          f"({(len(donor) - len(shared)) / len(donor):.1%} of donor)")


def main() -> None:
    print("## loading vocabs as raw-byte sets")
    olmo = hf_vocab_bytes(OLMO)
    deepseek = hf_vocab_bytes(DEEPSEEK)
    kimi = tiktoken_vocab_bytes(KIMI_TIKTOKEN)
    report("DeepSeek-V4-Flash", olmo, deepseek)
    report("Kimi-K2.6", olmo, kimi)


if __name__ == "__main__":
    main()

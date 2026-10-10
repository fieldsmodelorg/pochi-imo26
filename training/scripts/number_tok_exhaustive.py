"""Exhaustive check that OLMo3 / DeepSeek / Kimi segment numbers identically.

Compares the *decoded piece sequence* (segmentation), which is vocab-id-agnostic
and directly reflects the pretokenizer + merge behavior on digits. Reports the
first few mismatches per bucket, or confirms full agreement.
"""

from __future__ import annotations

import base64
import random

import tiktoken
from transformers import AutoTokenizer

OLMO = "/home/4/uq06834/Working/hdd/BigModels/Olmo-3-1025-7B"
DEEPSEEK = "/home/4/uq06834/Working/hdd/BigModels/DeepSeek-V4-Flash"
KIMI_TIKTOKEN = "/home/4/uq06834/Working/hdd/BigModels/Kimi-K2.6-tok/tiktoken.model"

KIMI_PAT = "|".join([
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


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


def main() -> None:
    olmo = AutoTokenizer.from_pretrained(OLMO, trust_remote_code=True)
    deepseek = AutoTokenizer.from_pretrained(DEEPSEEK, trust_remote_code=True)
    kimi = load_kimi()

    def seg_olmo(s):  # decoded piece sequence
        return tuple(olmo.decode([i]) for i in olmo.encode(s, add_special_tokens=False))

    def seg_ds(s):
        return tuple(deepseek.decode([i]) for i in deepseek.encode(s, add_special_tokens=False))

    def seg_kimi(s):
        return tuple(kimi.decode([i]) for i in kimi.encode(s))

    def run_bucket(name, cases):
        mismatches = []
        for s in cases:
            a, b, c = seg_olmo(s), seg_ds(s), seg_kimi(s)
            if not (a == b == c):
                mismatches.append((s, a, b, c))
        status = "ALL IDENTICAL" if not mismatches else f"{len(mismatches)} MISMATCHES"
        print(f"[{name}] {len(cases)} cases -> {status}")
        for s, a, b, c in mismatches[:8]:
            print(f"   {s!r}: OLMo={a} DS={b} Kimi={c}")
        return len(mismatches)

    total = 0
    # 1. every integer 0..10000
    total += run_bucket("int 0..10000", [str(i) for i in range(10001)])
    # 2. random big integers up to 30 digits
    rng = random.Random(0)
    total += run_bucket("rand big ints (5000, up to 30 digits)",
                        [str(rng.randrange(10 ** rng.randint(4, 30))) for _ in range(5000)])
    # 3. zero-led / repeated / boundary digit runs
    total += run_bucket("digit-run boundaries",
                        ["0" * n for n in range(1, 12)] +
                        ["1" + "0" * n for n in range(1, 12)] +
                        ["9" * n for n in range(1, 12)] +
                        [str(10 ** n) for n in range(0, 25)])
    # 4. decimals / math expressions
    total += run_bucket("decimals & math",
                        ["3.14159265358979", "2.718281828", "0.001", "1.5e10",
                         "1/3", "22/7", "x^2+y^2=z^2", "10^9+7", "998244353",
                         "1,000,000", "$1000", "1000th", "v2.0.1",
                         "a1b2c3", "  1000  ", "1000\n2000", "−1000", "1000kg"])
    # 5. with surrounding text (leading-space variants)
    total += run_bucket("in-context numbers",
                        [f"value {i}" for i in (1, 10, 100, 1000, 10000, 100000)] +
                        [f"={i}" for i in (1, 10, 100, 1000, 10000)] +
                        [f"step{i}" for i in (1, 12, 123, 1234)])

    print("\n" + ("=" * 50))
    print("OVERALL:", "ALL NUMBER TOKENIZATION IDENTICAL across OLMo3/DeepSeek/Kimi"
          if total == 0 else f"{total} TOTAL MISMATCHES FOUND")


if __name__ == "__main__":
    main()

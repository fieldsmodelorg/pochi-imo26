# Copyright 2026 proof-pilot. Apache-2.0.
"""Pre-render L2 docs to token ids (JSON) for the in-container validation/bench scripts.

Decouples L3 rendering (needs pyarrow + repo modules, runs in .venv-sglang) from the
sglang engine runs (run inside the lmsysorg/sglang apptainer image with minimal deps).

  .venv-sglang/bin/python training/teacher_extract/_render_docs.py \
      --out training/teacher_extract/_docs.json --n-docs 64 --max-doc-tokens 16384
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train_core"))

import pyarrow.parquet as pq

MODEL = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
SHARD = (os.environ.get("SFT_MIX", "data/nemotron-deepseek-sft-mix") + "/"
         "dataset=nemotron-math-v3/domain=aops_cot/part-00079.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--min-doc-tokens", type=int, default=256)
    ap.add_argument("--max-doc-tokens", type=int, default=16384)
    a = ap.parse_args()

    # AutoTokenizer can't parse the deepseek_v4 model config on transformers 5.x;
    # the tokenizer itself is a plain PreTrainedTokenizerFast.
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL)
    from l3_render import render_and_mask

    pf = pq.ParquetFile(SHARD)
    docs = []
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["id", "messages", "tools"])
        for i in range(tbl.num_rows):
            msgs = json.loads(tbl["messages"][i].as_py())
            tools_raw = tbl["tools"][i].as_py()
            tools = json.loads(tools_raw) if tools_raw else None
            rendered, _why = render_and_mask(msgs, tools, tokenizer, check_roundtrip=True)
            if rendered is None:
                continue
            if not (a.min_doc_tokens <= len(rendered.input_ids) <= a.max_doc_tokens):
                continue
            # distill needs teacher hidden at positions p whose NEXT token carries loss:
            # L3 labels are per-position (labels[t] = input_ids[t] or IGNORE, HF-shift
            # happens in the model), so p is a target iff labels[p+1] != IGNORE.
            from l3_render import IGNORE
            ids, labels = rendered.input_ids, rendered.labels
            positions = [p for p in range(len(ids) - 1) if labels[p + 1] != IGNORE]
            docs.append({"id": tbl["id"][i].as_py(), "input_ids": ids,
                         "positions": positions,
                         "targets": [ids[p + 1] for p in positions]})
            if len(docs) == a.n_docs:
                break
        if len(docs) == a.n_docs:
            break

    assert len(docs) >= 8, f"only found {len(docs)} suitable docs"
    with open(a.out, "w") as f:
        json.dump(docs, f)
    lens = sorted(len(d["input_ids"]) for d in docs)
    print(f"wrote {len(docs)} docs to {a.out}; tokens total={sum(lens):,} "
          f"min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")


if __name__ == "__main__":
    main()

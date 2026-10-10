# Copyright 2026 proof-pilot. Apache-2.0.
"""rollout dump tests (pure CPU, no GPU, seconds).

Verifies:
1. append->close produces parquet with correct row count/columns/types, loss_mask = [F]*plen+[T]*gen, input_ids==ids.
2. **dflash zero-conversion consumption**: read back exactly as dflash data.py loads (glob **/*.parquet ->
   datasets.load_dataset), input_ids/loss_mask are retrievable and can rebuild the next-token target (loss_mask only covers the completion).
3. **resume does not overwrite**: a second writer in the same directory continues the part numbering, total rows = the sum of both.
4. flush_interval periodic flush (writes even below a full batch).
5. enabled=False / dump=None: the produce path doesn't blow up (append is a no-op).

Run: PYTHONPATH=src .venv/bin/python tests/test_rollout_dump.py
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from opd_v2.rollout_store import RolloutDumpWriter


def _mk_rollouts(n, base=0):
    """Build n fake rollouts: (ids, prompt_len, wv, meta). Lengths/plen vary to verify loss_mask."""
    out = []
    for i in range(n):
        k = base + i
        plen = 3 + (k % 5)
        glen = 2 + (k % 7)
        ids = list(range(100 + k, 100 + k + plen + glen))   # arbitrary but verifiable token ids
        out.append((ids, plen, 10 + (k % 3), {"problem_id": f"p{k}", "template": "t"}))
    return out


def _read_all_parquet(d):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(d, "part-*.parquet")))
    assert files, f"no parquet files in {d}"
    tabs = [pq.read_table(f) for f in files]
    import pyarrow as pa
    return pa.concat_tables(tabs), files


async def _run_writer(d, rollouts, **kw):
    w = RolloutDumpWriter(d, **kw)
    w.start()
    for (ids, plen, wv, meta) in rollouts:
        w.append(ids, plen, wv, meta)
    await w.close()
    return w


def test_basic_and_schema():
    with tempfile.TemporaryDirectory() as d:
        rollouts = _mk_rollouts(25)
        w = asyncio.run(_run_writer(d, rollouts, rows_per_file=10, flush_interval_s=3600))
        assert w.n_written == 25, w.n_written
        # 25 rollouts / 10 per file -> 3 files (10+10+5)
        tab, files = _read_all_parquet(d)
        assert len(files) == 3, files
        assert tab.num_rows == 25
        names = set(tab.column_names)
        assert {"input_ids", "loss_mask", "wv", "prompt_len", "seq", "meta"} <= names, names
        import pyarrow as pa
        assert tab.schema.field("input_ids").type == pa.list_(pa.int32())
        assert tab.schema.field("loss_mask").type == pa.list_(pa.bool_())
        # per-row verification (sort by seq, map back to the original rollout)
        rows = tab.to_pylist()
        rows.sort(key=lambda r: r["seq"])
        for r, (ids, plen, wv, meta) in zip(rows, rollouts):
            assert r["input_ids"] == ids
            assert r["prompt_len"] == plen
            assert r["wv"] == wv
            assert r["loss_mask"] == [False] * plen + [True] * (len(ids) - plen)
            assert json.loads(r["meta"])["problem_id"] == meta["problem_id"]
        # dataset_info.json
        info = json.load(open(os.path.join(d, "dataset_info.json")))
        assert info["n_rollouts"] == 25 and info["n_files"] == 3
        print(f"[basic] OK: {w.n_written} rows, {len(files)} files, schema dflash-native + analysis cols")


def test_dflash_load_path():
    """Read back exactly as dflash data.py:load_dataset does, proving zero-conversion consumption."""
    from datasets import load_dataset as hf_load
    with tempfile.TemporaryDirectory() as d:
        rollouts = _mk_rollouts(13)
        asyncio.run(_run_writer(d, rollouts, rows_per_file=5, flush_interval_s=3600))
        # dflash data.py: glob('**/*.parquet') -> load_dataset('parquet', data_files=files)['train']
        files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
        ds = hf_load("parquet", data_files=files)["train"]
        assert len(ds) == 13, len(ds)
        assert "input_ids" in ds.column_names and "loss_mask" in ds.column_names
        # rebuild dflash's next-token supervision (only compute at loss_mask=True positions) - proves the semantics are usable
        s = ds[0]
        ii, lm = s["input_ids"], s["loss_mask"]
        assert len(ii) == len(lm)
        n_train = sum(1 for x in lm if x)
        assert n_train == len(ii) - s["prompt_len"]   # only the completion segment is in loss
        print(f"[dflash] OK: load_dataset read back {len(ds)} rows, input_ids/loss_mask consumable, loss covers completion {n_train} tok")


def test_resume_no_overwrite():
    with tempfile.TemporaryDirectory() as d:
        asyncio.run(_run_writer(d, _mk_rollouts(8, base=0), rows_per_file=5, flush_interval_s=3600))
        files1 = sorted(glob.glob(os.path.join(d, "part-*.parquet")))
        # second writer, same directory: should continue numbering, not overwrite
        asyncio.run(_run_writer(d, _mk_rollouts(6, base=100), rows_per_file=5, flush_interval_s=3600))
        files2 = sorted(glob.glob(os.path.join(d, "part-*.parquet")))
        assert len(files2) > len(files1), (files1, files2)
        assert set(files1) <= set(files2), "old files were overwritten"
        tab, _ = _read_all_parquet(d)
        assert tab.num_rows == 14, tab.num_rows   # 8 + 6
        print(f"[resume] OK: {len(files1)}->{len(files2)} files, no overwrite, {tab.num_rows} rows total")


def test_time_flush():
    """flush_interval expiry should flush to disk (even below a full batch)."""
    async def go(d):
        w = RolloutDumpWriter(d, rows_per_file=10000, flush_interval_s=1.0)
        w.start()
        for (ids, plen, wv, meta) in _mk_rollouts(3):
            w.append(ids, plen, wv, meta)
        await asyncio.sleep(2.5)            # wait for at least one flush tick
        n_mid = w.n_files
        await w.close()
        return n_mid, w.n_written
    with tempfile.TemporaryDirectory() as d:
        n_mid, n_written = asyncio.run(go(d))
        assert n_mid >= 1, f"time-flush did not fire (n_files={n_mid})"
        assert n_written == 3
        print(f"[time_flush] OK: flush_interval triggered a flush ({n_mid} files before close)")


def test_steady_load_flush():
    """Under steady inflow (queue never idle) it must still flush every flush_interval (regression: the old
    version only relied on the queue-idle _TICK, so under steady inflow it never periodically flushed and
    only waited for rows_per_file)."""
    async def go(d):
        w = RolloutDumpWriter(d, rows_per_file=100000, flush_interval_s=1.0)  # row cap high enough to never trigger
        w.start()
        # append one every 0.2s for ~3s (interval < flush_interval=1s -> the queue never idles for 1s)
        for _ in range(15):
            w.append(list(range(10, 18)), 3, 0, {})
            await asyncio.sleep(0.2)
        n_mid = w.n_files          # there should be files before close (periodic flush works)
        await w.close()
        return n_mid, w.n_written
    with tempfile.TemporaryDirectory() as d:
        n_mid, n_written = asyncio.run(go(d))
        assert n_mid >= 2, f"no periodic flush under steady inflow (n_files={n_mid} before close) = regression bug"
        assert n_written == 15
        print(f"[steady_load] OK: periodic flush works under steady inflow ({n_mid} files before close)")


def test_disabled_noop():
    """When dump=None, produce's append call must not blow up (here we directly verify append on a not-started writer is safe)."""
    w = RolloutDumpWriter(tempfile.mkdtemp(), rows_per_file=10)
    w.append([1, 2, 3], 1, 0, {})          # not started -> no-op, doesn't blow up
    print("[disabled] OK: append on a not-started writer is a safe no-op")


if __name__ == "__main__":
    test_basic_and_schema()
    test_dflash_load_path()
    test_resume_no_overwrite()
    test_time_flush()
    test_steady_load_flush()
    test_disabled_noop()
    print("\nALL ROLLOUT-DUMP TESTS PASSED")

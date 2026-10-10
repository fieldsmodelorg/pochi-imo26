# Copyright 2026 proof-pilot. Apache-2.0.
"""rollout dump — write **all** student rollouts (prompt+response token ids) to dflash-native parquet.

Why store them: the rollout token ids of on-policy distillation only ever live in memory
(producer->buffer->HTTP->trainer, discarded after use) and are nowhere on disk (the teacher hidden `.bin`
doesn't contain token ids and is GC'd after use). But these rollouts are valuable for (a) post-hoc analysis
and (b) spec-decode draft (dflash) training. This module writes each **successfully generated** rollout to
parquet as a side channel, **fully decoupled** from teacher-scoring success/failure, buffer
overflow/staleness, and the hidden GC.

Format = **dflash `train.py` native** (zero-conversion consumption; dflash `data.py` reads by column name
`input_ids`/`loss_mask` and ignores extra columns, see dflash data.py:302/305):
  input_ids : list<int32>  = traj.ids (prompt+response, already truncated to max_traj_tokens)
  loss_mask : list<bool>   = [False]*prompt_len + [True]*gen_len (prompt not in loss, completion is)
  # ---- the following are analysis columns (dflash ignores; for post-hoc analysis/dedup/staleness studies) ----
  wv         : int32       the rollout weight_version that generated this trajectory
  prompt_len : int32
  seq        : int64       globally monotonic sequence number (this process)
  meta       : string(JSON) problem_id/template etc. (only when store_meta=True)

Persistence strategy (single writer = orchestrator single process, single event loop):
- `append()` only enqueues (microseconds, non-blocking, on the event loop); a background coroutine pulls
  batches and `run_in_executor` writes files on a thread -> the event loop never blocks on disk I/O. append
  only does `put_nowait`+counter, atomic within the single event loop, safe for 64 concurrent produce atoms.
- **Each flush = one complete standalone parquet file** (tmp->atomic rename; footer written immediately) -> a
  crash only loses "the not-yet-flushed in-memory batch" (≤ rows_per_file or ≤ flush_interval_s), no
  long-lived ParquetWriter (avoids an open writer trashing a whole shard on crash). dflash globs
  `**/*.parquet` over the directory -> multiple small files is fine.
- On startup, scan existing `part-*.parquet` and continue numbering (resume/rerun does not overwrite).
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re

log = logging.getLogger("opd_v2.rollout_store")

_STOP = object()       # writer shutdown sentinel
_TICK = object()       # flush_interval timeout sentinel (triggers periodic flush)
_PART_RE = re.compile(r"part-(\d+)\.parquet$")


class RolloutDumpWriter:
    """Write rollout token ids to dflash-native parquet as a side channel (single writer, non-blocking, complete file per flush)."""

    def __init__(self, dump_dir: str, *, rows_per_file: int = 1000,
                 flush_interval_s: float = 60.0, store_meta: bool = True,
                 compression: str = "zstd", provenance: dict | None = None):
        self.dir = dump_dir
        self.rows_per_file = max(1, int(rows_per_file))
        self.flush_interval_s = max(1.0, float(flush_interval_s))
        self.store_meta = bool(store_meta)
        self.compression = compression
        self.provenance = dict(provenance or {})
        os.makedirs(self.dir, exist_ok=True)
        # resume: continue from the existing max part index, do not overwrite
        self._file_idx = self._scan_next_idx()
        self._seq = 0
        # stats (event loop reads, writer thread writes; all int/float, stats is observation-only, tolerates looseness)
        self.n_written = 0
        self.n_files = 0
        self.n_bytes = 0
        self.queue_high = 0
        self._q: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._schema = None   # deferred to inside the writer thread (avoid importing pyarrow during event-loop import)

    def _scan_next_idx(self) -> int:
        mx = -1
        for p in glob.glob(os.path.join(self.dir, "part-*.parquet")):
            m = _PART_RE.search(os.path.basename(p))
            if m:
                mx = max(mx, int(m.group(1)))
        return mx + 1

    # ---- lifecycle (called inside a running event loop) ----
    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._q = asyncio.Queue()
        self._task = asyncio.create_task(self._run(), name="rollout_dump_writer")
        log.info("rollout dump -> %s (rows/file=%d, flush=%.0fs, meta=%s, next_part=%d)",
                 self.dir, self.rows_per_file, self.flush_interval_s, self.store_meta, self._file_idx)

    def append(self, ids: list[int], prompt_len: int, wv: int, meta: dict | None = None) -> None:
        """Non-blocking enqueue of one rollout. `ids` = prompt+response (already truncated). Safe within the single event loop."""
        if self._q is None:
            return
        seq = self._seq
        self._seq += 1
        # store the ids reference (no copy); convert meta to compact JSON up front (skipped when store_meta=False)
        meta_json = json.dumps(meta, ensure_ascii=False, separators=(",", ":")) if (self.store_meta and meta) else ""
        self._q.put_nowait((seq, int(wv), int(prompt_len), ids, meta_json))
        qs = self._q.qsize()
        if qs > self.queue_high:
            self.queue_high = qs
            if qs and qs % 5000 == 0:
                log.warning("rollout dump queue high-water=%d (writer falling behind?)", qs)

    async def close(self) -> None:
        """Teardown: stop the writer, flush remaining, write dataset_info.json. Idempotent."""
        if self._task is None:
            return
        self._q.put_nowait(_STOP)
        try:
            await self._task
        except Exception:
            log.exception("rollout dump writer task error on close")
        self._task = None
        self._write_dataset_info()
        log.info("rollout dump closed: %d rollouts -> %d files (%.2f GB)",
                 self.n_written, self.n_files, self.n_bytes / 1e9)

    # ---- background writer ----
    async def _run(self) -> None:
        batch: list = []
        last_flush = self._loop.time()
        # poll interval = min(flush_interval, 5s): even when rollouts stream in steadily (queue never idle), we
        # still flush every flush_interval **by wall-clock** (fixes the "only _TICK when the queue is idle" ->
        # never periodically flushes under steady inflow bug).
        poll = min(self.flush_interval_s, 5.0)
        while True:
            try:
                item = await asyncio.wait_for(self._q.get(), timeout=poll)
            except asyncio.TimeoutError:
                item = _TICK
            if item is _STOP:
                break
            if item is not _TICK:
                batch.append(item)
            now = self._loop.time()
            # full batch -> flush immediately; otherwise if flush_interval has passed since the last flush and there is accumulation -> periodic flush
            if batch and (len(batch) >= self.rows_per_file or now - last_flush >= self.flush_interval_s):
                await self._flush(batch)
                batch = []
                last_flush = now
        # teardown: drain the queue then do a final flush
        while self._q is not None and not self._q.empty():
            it = self._q.get_nowait()
            if it not in (_STOP, _TICK):
                batch.append(it)
            if len(batch) >= self.rows_per_file:
                await self._flush(batch)
                batch = []
        if batch:
            await self._flush(batch)

    async def _flush(self, batch: list) -> None:
        if not batch:
            return
        try:
            await self._loop.run_in_executor(None, self._write_file, batch)
        except Exception:
            log.exception("rollout dump flush failed (%d rows dropped)", len(batch))

    def _write_file(self, batch: list) -> None:
        """Write one complete parquet file on the executor thread (tmp->atomic rename)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        if self._schema is None:
            fields = [("input_ids", pa.list_(pa.int32())), ("loss_mask", pa.list_(pa.bool_())),
                      ("wv", pa.int32()), ("prompt_len", pa.int32()), ("seq", pa.int64())]
            if self.store_meta:
                fields.append(("meta", pa.string()))
            self._schema = pa.schema(fields)

        input_ids, loss_mask, wv, plen, seq, meta = [], [], [], [], [], []
        for (s, w, pl, ids, mj) in batch:
            input_ids.append(ids)
            loss_mask.append([False] * pl + [True] * (len(ids) - pl))
            wv.append(w); plen.append(pl); seq.append(s); meta.append(mj)
        cols = {"input_ids": pa.array(input_ids, type=pa.list_(pa.int32())),
                "loss_mask": pa.array(loss_mask, type=pa.list_(pa.bool_())),
                "wv": pa.array(wv, type=pa.int32()),
                "prompt_len": pa.array(plen, type=pa.int32()),
                "seq": pa.array(seq, type=pa.int64())}
        if self.store_meta:
            cols["meta"] = pa.array(meta, type=pa.string())
        table = pa.table(cols, schema=self._schema)

        idx = self._file_idx
        self._file_idx += 1
        final = os.path.join(self.dir, f"part-{idx:08d}.parquet")
        tmp = final + ".tmp"
        pq.write_table(table, tmp, compression=self.compression)
        os.replace(tmp, final)            # atomic: dflash glob never picks up a half-written file
        self.n_written += len(batch)
        self.n_files += 1
        try:
            self.n_bytes += os.path.getsize(final)
        except OSError:
            pass

    def _write_dataset_info(self) -> None:
        info = {
            "format": "dflash-native parquet",
            "columns": {"input_ids": "list<int32>", "loss_mask": "list<bool>",
                        "wv": "int32", "prompt_len": "int32", "seq": "int64",
                        **({"meta": "string(JSON)"} if self.store_meta else {})},
            "n_rollouts": self.n_written,
            "n_files": self.n_files,
            "n_bytes": self.n_bytes,
            "source": "opd_v2 on-policy distillation rollouts",
            **self.provenance,
        }
        try:
            with open(os.path.join(self.dir, "dataset_info.json"), "w") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception:
            log.exception("rollout dump dataset_info write failed")

    def stats(self) -> dict:
        return {"n_written": self.n_written, "n_files": self.n_files,
                "n_bytes": self.n_bytes,
                "queue": (self._q.qsize() if self._q is not None else 0),
                "queue_high": self.queue_high}

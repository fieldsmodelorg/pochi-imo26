# Copyright 2026 proof-pilot. Apache-2.0.
"""shared-FS spool for teacher hidden — v2's core transport fix (fixes v1 P7).

v1 sent teacher hidden (a single 100k traj = 332MB) over HTTP bytes -> orchestrator RAM -> gloo scatter of
the whole pickle to every rank, the biggest waste in the pipeline. v2 instead:

    teacher computes hidden -> **server-side writes it into shared FS (WekaFS /work)** -> returns only a
    handle (tens of bytes) -> orchestrator/buffer/scatter only ever move handles -> **the owning trainer
    rank reads it from FS itself**.

So the physical path of the bytes is teacher -> FS -> trainer rank point-to-point, **never through the
orchestrator, never through gloo**.

This file provides:
- **File format**: a 64-byte self-describing header + payload (`packed || scales || top1`). The file
  carries its own layout, so the reader does not depend on the handle's metadata (avoids the v1
  header/handle drift bug).
- `HiddenHandle`: on the orchestrator side you only need `{path, seq_len, wv}` (seq_len for buffer token
  accounting, wv for re-validating staleness, path for reading + GC).
- `write_hidden` / `read_hidden`: teacher-side write (atomic: tmp + rename), trainer-side read.
- `HiddenStore`: manages the hidden directory, generates unique paths, GC (unlink + background TTL backstop, V14).

torch-free (the orchestrator, a pure-CPU process, must be able to import it); the payload is raw bytes,
decode happens on the trainer GPU side.
"""
from __future__ import annotations

import os
import struct
import time
import uuid
from dataclasses import dataclass

from opd_v2.config import HID_DIM

# header: magic(4) version(4) seq_len(4) hid(4) packed_len(8) scales_len(8) top1_len(8) = 40, padded to 64.
_MAGIC = b"OPDH"
_VERSION = 1
_HEADER_FMT = "<4sIIIQQQ"
_HEADER_CORE = struct.calcsize(_HEADER_FMT)   # 40
HEADER_SIZE = 64


def packed_row_bytes(hid: int = HID_DIM) -> int:
    return hid * 6 // 8


def scale_row_bytes(hid: int = HID_DIM) -> int:
    return (hid // 32) * 2


@dataclass
class HiddenHandle:
    """The small object the orchestrator moves (no bytes). The file is self-describing; this only keeps what GC/accounting needs."""
    path: str          # shared-FS absolute path
    seq_len: int       # number of positions the teacher returned (= number of rows for alignment); for buffer token accounting
    wv: int            # the weight_version when the rollout was generated (staleness can be re-validated on read)

    def to_dict(self) -> dict:
        return {"path": self.path, "seq_len": self.seq_len, "wv": self.wv}

    @classmethod
    def from_dict(cls, d: dict) -> "HiddenHandle":
        return cls(path=d["path"], seq_len=int(d["seq_len"]), wv=int(d["wv"]))


def write_hidden(path: str, packed: bytes, scales: bytes, seq_len: int,
                 top1: bytes = b"", hid: int = HID_DIM) -> None:
    """Atomically write a hidden file (called server-side by the teacher). Write `<path>.tmp.<pid>` then rename.

    Validates that the payload lengths are consistent with seq_len/hid (catch codec/slicing bugs early,
    better than a mysterious reshape blowup on the trainer side).
    """
    exp_p = seq_len * packed_row_bytes(hid)
    exp_s = seq_len * scale_row_bytes(hid)
    if len(packed) != exp_p:
        raise ValueError(f"packed len {len(packed)} != seq_len*pcols {exp_p} (seq_len={seq_len} hid={hid})")
    if len(scales) != exp_s:
        raise ValueError(f"scales len {len(scales)} != seq_len*scols {exp_s} (seq_len={seq_len} hid={hid})")
    if top1 and len(top1) != seq_len * 4:
        raise ValueError(f"top1 len {len(top1)} != seq_len*4 {seq_len * 4}")
    header = struct.pack(_HEADER_FMT, _MAGIC, _VERSION, seq_len, hid,
                         len(packed), len(scales), len(top1))
    header = header + b"\x00" * (HEADER_SIZE - _HEADER_CORE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(packed)
        f.write(scales)
        if top1:
            f.write(top1)
    os.replace(tmp, path)   # atomic on POSIX; the reader always sees a complete file or nothing


def read_hidden(path: str) -> tuple[bytes, bytes, bytes, int, int]:
    """Read a hidden file (called by the owning trainer rank). Returns (packed, scales, top1, seq_len, hid).

    The file carries its own layout (does not rely on the handle); on error (GC'd / orphan / half-written) it
    raises FileNotFoundError/ValueError, which the trainer's collective-safe gate (§6.5) turns into an
    all-ranks skip.
    """
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            raise ValueError(f"hidden file too short for header: {path}")
        magic, ver, seq_len, hid, plen, slen, tlen = struct.unpack(_HEADER_FMT, head[:_HEADER_CORE])
        if magic != _MAGIC:
            raise ValueError(f"bad hidden magic {magic!r} in {path}")
        if ver != _VERSION:
            raise ValueError(f"unsupported hidden version {ver} in {path}")
        packed = f.read(plen)
        scales = f.read(slen)
        top1 = f.read(tlen) if tlen else b""
    if len(packed) != plen or len(scales) != slen or len(top1) != tlen:
        raise ValueError(f"hidden payload truncated: {path}")
    return packed, scales, top1, int(seq_len), int(hid)


class HiddenStore:
    """Manage one run's hidden directory (shared FS): generate unique paths, GC (unlink + TTL backstop).

    owner = orchestrator (it creates handles, tracks staleness, sends to the trainer, so it owns deletion, V14).
    """

    def __init__(self, hidden_dir: str):
        self.dir = hidden_dir
        os.makedirs(self.dir, exist_ok=True)

    def new_path(self) -> str:
        """Generate a hidden-file path unique within this run (the client creates it before calling the teacher, passes it for the teacher to write, and remembers it for GC)."""
        return os.path.join(self.dir, f"{uuid.uuid4().hex}.bin")

    def delete(self, path: str) -> bool:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def delete_handles(self, handles) -> int:
        """Delete a batch (after /train_step returns / after stale-drop). Returns the number actually deleted."""
        n = 0
        for h in handles:
            p = h.path if isinstance(h, HiddenHandle) else h
            if p and self.delete(p):
                n += 1
        return n

    def sweep_ttl(self, ttl_seconds: float) -> int:
        """Background TTL sweep: clean orphans left by an orchestrator crash (mtime older than ttl). Returns the number cleaned.

        run-scoped (only sweeps this run's hidden_dir), so it won't wrongly delete another run's files.
        """
        now = time.time()
        n = 0
        try:
            entries = os.listdir(self.dir)
        except FileNotFoundError:
            return 0
        for name in entries:
            if not name.endswith(".bin"):
                continue
            p = os.path.join(self.dir, name)
            try:
                if now - os.path.getmtime(p) > ttl_seconds:
                    if self.delete(p):
                        n += 1
            except OSError:
                pass
        return n

    def usage(self) -> tuple[int, int]:
        """(number of files, total bytes) — for wandb observation."""
        n, total = 0, 0
        try:
            for name in os.listdir(self.dir):
                if name.endswith(".bin"):
                    try:
                        total += os.path.getsize(os.path.join(self.dir, name))
                        n += 1
                    except OSError:
                        pass
        except FileNotFoundError:
            pass
        return n, total

# Copyright 2026 proof-pilot. Apache-2.0.
"""Stage-1 SFT entrypoint for the FMI Singularity *train variant* (v2, single-source layout).

v2 of `training/stage1/` (archived). The trainer is the validated stage-1 trainer, ported
unchanged except for imports: shared modules are imported from their canonical top-level
homes (`olmo3_sink/`, `train_core/`) instead of a vendored snapshot. The container still
gets a flat self-contained /app -- materialized at packaging time by `make_pkg.py` from
`pkg.manifest` (see ../README.md).

Trains `Olmo-3-7B-Think-deepseekTok` on the L2 SFT mix (`nemotron-deepseek-sft-mix`)
with the olmo3_sink architecture (learnable attention sink + FA3 packing-metadata reuse).

FMI train-variant CLI contract (sole entrypoint `/app/train.py`):
    --model_path     local path to the base model weights (Olmo-3-7B-Think-deepseekTok)
    --dataset_path   local path to the L2 parquet (hive: dataset=*/domain=*)
    --output_path    where trained weights are written
    --logdir         where logs are written
plus the optional FMI-named knobs (--num_gpus/--learning_rate/--num_train_epochs/
--per_device_batch_size/--gradient_accumulation_steps) and our own (--micro-len etc.).

Pipeline per rank: stream L2 parquet shards (disjoint across ranks) -> L3 render+mask
(`l3_render`) -> shuffle buffer -> greedy length-packing into fixed `micro_len` rows
(`PackedCollator`) -> fwd/CE/backward with grad-accum -> cosine-LR optimizer step ->
periodic sharded (DCP) checkpoint for resume + consolidated HF save at the end.

Parallelism is torchrun-env driven:
  - 1 GPU            : plain single-device, PagedAdamW8bit (optimizer state on CPU).
  - 1 node, N GPUs   : FSDP2 full-shard (params+grads+optim sharded), fused AdamW.
  - M nodes x N GPUs : HSDP -- FSDP2 full-shard *within* a node, replicate *across* nodes
                       (a 2-D device mesh). Grads all-reduce across nodes, params/optim
                       all-gather stays on NVLink. Single-node is the degenerate 1-D case
                       of the same code path. The 3x8 H200 target is M=3, N=8.

See training/stage1/README.md and repo docs/train_bench.md for the throughput recipe.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

# Dual-context import resolution (no try/except fallbacks -- fail loud):
#   - repo:      walk up to the `pyproject.toml` marker and put the repo root on sys.path,
#                so `olmo3_sink`/`train_core` resolve to the canonical top-level packages.
#   - container: /app/train.py has no pyproject.toml above it; the loop is a no-op and
#                PYTHONPATH=/app (set in the .def %environment) provides the materialized
#                copies. A missing package then raises ImportError immediately.
for _p in Path(__file__).resolve().parents:
    if (_p / "pyproject.toml").is_file():
        sys.path.insert(0, str(_p))
        break

import pyarrow.parquet as pq
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo3_sink import Olmo3SinkConfig, apply_liger, register_olmo3_sink
from olmo3_sink.sft_data import Example, greedy_pack, pack_to_tensors
from train_core.l3_render import IGNORE, render_and_mask

# ---- CONSTANTS ----
ATTN_IMPL = "olmo3_sink_fa3"  # in-kernel FA3 sink (container builds patched FA3); "eager" is the no-FA3 reference
SINK_INIT = 0.0               # learnable per-head sink logit init -- ONLY fills sinks missing from
                              # the checkpoint. Post-mortem of both stage-1 runs (10k steps each):
                              # init 0.0 is in practice a DEAD start (sinks moved <=0.06, absorb
                              # 0.07% -- the probe's "250x larger gradient than -10" is real but
                              # still ~6.5 nats below the drafted-token dump level; see
                              # docs/attn_sink_study.md §9). Preferred path: a fused checkpoint
                              # from olmo3_sink.build_init_model with measured per-head warm-start
                              # sinks BAKED into the weights (this constant is then irrelevant).
SHUFFLE_BUF = 256             # examples per length-packing window (also the shuffle window)
PREFETCH_BINS = 16            # packed bins kept ready in the background prefetch queue


# ---- distributed / logging ----
def setup_distributed() -> tuple[int, int, int]:
    """Read torchrun env. Returns (rank, world_size, local_rank)."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        import datetime

        torch.cuda.set_device(local)
        # device_id binds the PG to this rank's device (enables NCCL comm-init and silences
        # the barrier() device-context warning); required well-formed for FSDP2 collectives.
        # timeout: the default 10-min NCCL watchdog killed the 32B 56-rank run during its
        # FIRST periodic DCP save (2026-06-05 job 79388, ALLREDUCE NumelIn=1 = the post-save
        # barrier/step-sync): a ~390 GB fp32 master+optim checkpoint to WekaFS can keep
        # straggler ranks writing >10 min while finished ranks wait in the next collective.
        # 60 min tolerates giant saves; real hangs then take 1 h to surface, acceptable for
        # a checkpointed --requeue run.
        timeout = datetime.timedelta(minutes=int(os.environ.get("DIST_TIMEOUT_MIN", "60")))
        torch.distributed.init_process_group(
            "nccl", device_id=torch.device(f"cuda:{local}"), timeout=timeout)
    return rank, world, local


def setup_logging(logdir: str, rank: int) -> None:
    Path(logdir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(os.path.join(logdir, f"train_rank{rank}.log"), level="DEBUG", enqueue=True)
    if rank == 0:
        import sys
        logger.add(sys.stderr, level="INFO")


def is_main(rank: int) -> bool:
    return rank == 0


def setup_wandb(project: Optional[str], run_name: Optional[str], mode: str, logdir: str,
                rank: int, config: dict):
    """Init wandb on rank 0 only. Returns the run handle or None.

    Off unless --wandb-project is given. `mode` is online/offline/disabled; offline writes a
    local run under <logdir>/wandb that `wandb sync` can upload later (useful if a compute
    node has no egress). Lazy + non-fatal: a missing package or failed init logs a warning and
    training proceeds without wandb."""
    if not project or not is_main(rank):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; pip install wandb. Continuing without it.")
        return None
    try:
        run = wandb.init(project=project, name=run_name, mode=mode,
                         dir=logdir, config=config)
        logger.info(f"wandb: project={project} run={run.name} mode={mode} url={run.get_url() or '(offline)'}")
        return run
    except Exception as e:  # noqa: BLE001 -- logging must never kill training
        logger.warning(f"wandb.init failed ({type(e).__name__}: {e}); continuing without it.")
        return None


# ---- model ----
def build_model(model_path: str, *, attn: str = ATTN_IMPL, liger: str | bool = True,
                master_dtype: torch.dtype = torch.bfloat16):
    """Olmo3Sink model from a stock-Olmo3 checkpoint + learnable sink + Liger kernels.

    `master_dtype` is the dtype the weights are LOADED/kept in (the optimizer's master copy).
    On the FSDP path we load fp32 and rely on MixedPrecisionPolicy(param_dtype=bf16) to cast
    to bf16 for compute -> fp32 master + bf16 compute (avoids the bf16 "stale weights" lost-
    update problem; ~same speed, better convergence). bf16 keeps the old all-bf16 behaviour.
    Uses the throughput-validated olmo3_sink config (learnable sink + Liger kernels).
    """
    register_olmo3_sink()
    d = AutoConfig.from_pretrained(model_path).to_dict()
    d.pop("model_type", None)
    d.pop("architectures", None)
    cfg = Olmo3SinkConfig(**d)
    cfg.sink_init_value = SINK_INIT
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=cfg, dtype=master_dtype, attn_implementation=attn)
    # Record the sink init this run actually got. SINK_INIT only fills sinks MISSING from
    # the checkpoint; fused models (olmo3_sink.build_init_model: measured per-head warm
    # start baked into the weights -- docs/attn_sink_study.md §9/§10) ship their own and
    # must arrive intact here (canary for the tf5 _init_weights zeroing bug, fixed 20c49d8).
    with torch.no_grad():
        sk = torch.stack([l.self_attn.sinks.float() for l in model.model.layers])
    kind = ("baked warm-start" if sk.abs().max() > 1.0 else
            f"scalar init {SINK_INIT}: measured-dead for SFT injection (docs §9), "
            f"prefer a build_init_model fused checkpoint")
    logger.info(f"sinks at load: mean {sk.mean():+.3f} min {sk.min():+.3f} "
                f"max {sk.max():+.3f} ({kind})")
    if liger:
        if liger == "ce-only":
            apply_liger(model, rope=False, rms_norm=False, swiglu=False,
                        fused_linear_cross_entropy=True)
        else:
            apply_liger(model)
    model.config.use_cache = False
    return model


def setup_parallelism(model, world: int, local: int):
    """Place / shard the model.

    world==1            -> single device.
    world==gpus/node    -> FSDP2 full-shard over a 1-D mesh (single node).
    world > gpus/node   -> HSDP: 2-D mesh (replicate=n_nodes, shard=gpus/node); FSDP2
                           full-shard within a node, replicate across nodes.
    """
    if world <= 1:
        return model.to(f"cuda:{local}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count())) or 1
    if world % gpus_per_node != 0:
        raise ValueError(
            f"world_size {world} not divisible by gpus_per_node {gpus_per_node}; "
            "set LOCAL_WORLD_SIZE correctly (torchrun does this).")
    n_nodes = world // gpus_per_node

    if n_nodes > 1:
        mesh = init_device_mesh("cuda", (n_nodes, gpus_per_node),
                                mesh_dim_names=("replicate", "shard"))
        logger.info(f"HSDP device mesh: replicate={n_nodes} x shard={gpus_per_node}")
    else:
        mesh = init_device_mesh("cuda", (gpus_per_node,), mesh_dim_names=("shard",))
        logger.info(f"FSDP2 device mesh: shard={gpus_per_node}")

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model = model.to(f"cuda:{local}")
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)

    # Confirm FSDP2 actually engaged: after fully_shard the params are DTensors whose
    # local shard is ~1/shard_size of the global size (not replicated). Logged once so a
    # run's logs prove sharding happened (vs silently falling back to replicated/DDP).
    from torch.distributed.tensor import DTensor
    p = next(model.parameters())
    glob = sum(pp.numel() if not isinstance(pp, DTensor) else pp.to_local().numel() * mesh["shard"].size()
               for pp in model.parameters())
    locl = sum(pp.to_local().numel() if isinstance(pp, DTensor) else pp.numel() for pp in model.parameters())
    logger.info(f"FSDP2 engaged: DTensor={isinstance(p, DTensor)} placements={getattr(p, 'placements', None)} "
                f"| local params {locl/1e9:.2f}B / global {glob/1e9:.2f}B (shard={mesh['shard'].size()})")
    return model


# ---- data (L3: render + mask + pack) ----
def iter_examples(dataset_paths: list[Path], mix_entries: list[dict], tokenizer,
                  max_len: int, rank: int, world: int,
                  epoch: int = 0, shuffle: bool = True) -> Iterator[Example]:
    """Stream the weighted L2 mix -> rendered/masked/length-capped Examples (per rank).

    The shard task list comes from `data_mix.build_shard_tasks` (multi-root, per-partition
    `repeat` weights; repeat = k + f -> k full passes per shard + one pass keeping an
    f-fraction of rows via the epoch-sliding `keep_row` hash window -- see data_mix.py).
    Tasks are partitioned round-robin across ranks (disjoint, whole-task granularity).
    Per epoch the task order is permuted (seeded by epoch) so multi-epoch runs differ.
    With no mix entries and a single root this is exactly the v1 behaviour.

    Stage-1 is a foundation run that does not need very long context, so an example longer
    than `max_len` is **right-truncated** to `max_len` (keep the prompt + the start of the
    answer) rather than dropped -- this retains the long-tail data (which is ~14% of rows /
    ~40% of tokens at 32k, skewed toward the longest proofs) instead of discarding it. A
    truncated example loses its trailing EOS; that is fine because the many shorter examples
    still teach termination, and the truncated prefix still carries reasoning signal. If a
    truncation removes every assistant target token (the cut lands before any answer), the
    example is dropped (nothing to learn).
    """
    import random

    from data_mix import assign_tasks, build_shard_tasks, keep_row

    tasks = build_shard_tasks(dataset_paths, mix_entries, epoch)
    # Size-balanced (LPT) assignment, NOT index round-robin: task sizes span ~25x and the
    # rank-synced MIN-stop ends the epoch when the LIGHTEST rank runs dry -- round-robin
    # truncated epoch 0 at 13% on 64 ranks (job 79398). Within-rank order is epoch-shuffled.
    my_tasks = assign_tasks(tasks, world)[rank]
    if shuffle:
        random.Random(1234 + epoch).shuffle(my_tasks)
    logger.info(f"rank {rank}: {len(my_tasks)}/{len(tasks)} shard tasks (epoch {epoch})")

    kept = truncated = dropped = mix_skipped = render_errors = 0
    for task in my_tasks:
        pf = pq.ParquetFile(str(task.path))
        # the `id` column is only needed (and only read) for fractional-window tasks
        cols = ["messages", "tools"] + (["id"] if task.frac is not None else [])
        for batch in pf.iter_batches(batch_size=256, columns=cols):
            msgs_col, tools_col = batch.column(0), batch.column(1)
            id_col = batch.column(2) if task.frac is not None else None
            for i in range(batch.num_rows):
                if id_col is not None and not keep_row(id_col[i].as_py(), task.frac, epoch):
                    mix_skipped += 1   # outside this epoch's hash window
                    continue
                try:
                    msgs = json.loads(msgs_col[i].as_py())
                    traw = tools_col[i].as_py()
                    tools = json.loads(traw) if traw else None
                    # check_roundtrip=False in the hot path: the L3 render is verified offline
                    # by _l3_test.py; re-running encode_messages per row would ~2x tokenize cost.
                    rendered, _why = render_and_mask(msgs, tools, tokenizer, check_roundtrip=False)
                except Exception as e:  # noqa: BLE001 -- malformed upstream row
                    # Skip loudly, never die: a multi-million-row public mix has a tail of
                    # broken rows (observed 2026-06-05 job 79383: a row with a corrupted
                    # role string 'system\n{prompt_style.system_prompt}<|im_end|>' killed
                    # all 64 ranks at step ~455 via render_message's Unknown-role raise).
                    # Same policy as the L2 build's NUL-line skip: count + sample-log.
                    render_errors += 1
                    if render_errors <= 5:
                        logger.warning(f"render error in {task.path.name} batch-row {i}: "
                                       f"{type(e).__name__}: {e} -- row skipped")
                    continue
                if rendered is None:
                    continue
                ids, labels = rendered.input_ids, rendered.labels
                if len(ids) > max_len:
                    ids, labels = ids[:max_len], labels[:max_len]
                    if all(l == IGNORE for l in labels):
                        dropped += 1   # truncation removed all targets -> nothing to learn
                        continue
                    truncated += 1
                kept += 1
                yield Example(ids, labels=labels)
    logger.info(f"rank {rank}: epoch {epoch} done -- kept={kept} truncated(>{max_len})={truncated} "
                f"dropped(no-target-after-trunc)={dropped} mix_skipped={mix_skipped} "
                f"render_errors={render_errors}")


def l4_meta(dataset_paths: list[Path]) -> Optional[dict]:
    """If dataset_path is a single L4 directory (build_l4.py output), return its meta."""
    if len(dataset_paths) != 1 or not (dataset_paths[0] / "meta.json").exists():
        return None
    with open(dataset_paths[0] / "meta.json") as f:
        meta = json.load(f)
    return meta if meta.get("format") == "proof-pilot-l4-v1" else None


def iter_l4_bins(root: Path, meta: dict, rank: int, world: int, epoch: int,
                 max_segs: Optional[int]):
    """Stream this rank's bins of a pre-packed L4 dataset (offline render+shuffle+pack).

    Global bin order is a seeded permutation striped across ranks -> every rank gets
    floor/ceil(N/world) identical-cost bins (no imbalance, epoch ends in lockstep) and
    every bin is an iid sample of the mix (offline global row shuffle). Emits exactly
    the dict `pack_to_tensors` produces (input_ids/position_ids/labels [1,L] + n_docs
    + optional fixed-shape varlen metadata). NOTE: L4 materializes one fractional-
    window epoch; training epochs beyond the first reuse the same bins in a new order.
    """
    import numpy as np

    N, L, pad_id = meta["n_bins"], meta["micro_len"], meta["pad_id"]
    ids_mm = np.memmap(root / "input_ids.i32", dtype=np.int32, mode="r", shape=(N, L))
    msk_mm = np.memmap(root / "loss_mask.bits", dtype=np.uint8, mode="r", shape=(N, L // 8))
    seg_ptr = np.fromfile(root / "seg_ptr.i64", dtype=np.int64)
    seg_lens = np.fromfile(root / "seg_lens.i32", dtype=np.int32)
    # Cost-balanced step groups: bins are equal-token but attention cost ~ sum(len^2)
    # varies ~3x across bins; with one bin per rank per step and per-layer FSDP
    # collectives, every step runs at the cost of its most expensive bin (measured
    # 2.9x ideal on the lc256k run, 2026-06-12). Sort bins by cost, take consecutive
    # groups of `world` (cost-homogeneous steps), shuffle group order per epoch;
    # rank r consumes the r-th bin of each group. Identical on every rank; drops
    # the N % world cheapest-tail bins.
    costs = np.add.reduceat(seg_lens.astype(np.float64) ** 2, seg_ptr[:-1])
    by_cost = np.argsort(-costs, kind="stable")
    n_grp = N // world
    groups = by_cost[:n_grp * world].reshape(n_grp, world)
    gorder = np.random.RandomState(1234 + epoch).permutation(n_grp)
    order = groups[gorder, rank]
    logger.info(f"rank {rank}: L4 {len(order)}/{N} bins (epoch {epoch}, cost-balanced)")
    for j in order:
        j = int(j)
        row = ids_mm[j]
        mask = np.unpackbits(msk_mm[j], count=L).astype(bool)
        lens = seg_lens[seg_ptr[j]:seg_ptr[j + 1]]
        pos = np.concatenate([np.arange(l, dtype=np.int64) for l in lens])
        # trailing pad segment iff the last segment carries no loss (every real example
        # has >= 1 target token by construction -- no-target rows are dropped at build)
        padded = not mask[L - int(lens[-1]):].any()
        out = {
            "input_ids": torch.from_numpy(row.astype(np.int64))[None],
            "position_ids": torch.from_numpy(pos)[None],
            "labels": torch.from_numpy(
                np.where(mask, row, IGNORE).astype(np.int64))[None],
            "n_docs": int(len(lens)) - (1 if padded else 0),
        }
        if max_segs is not None:
            n = len(lens)
            assert n <= max_segs, f"L4 bin {j} has {n} segments > max_segs={max_segs}"
            cu = torch.zeros(max_segs + 1, dtype=torch.int32)
            cu[1:n + 1] = torch.from_numpy(np.cumsum(lens).astype(np.int32))
            cu[n + 1:] = L
            out["cu_seq_lens_q"] = out["cu_seq_lens_k"] = cu
            out["max_length_q"] = out["max_length_k"] = L
        yield out


def count_dataset_docs(dataset_paths: list[Path], mix_entries: list[dict]) -> float:
    """Repeat-weighted total docs of the mix, from parquet metadata only (no data read).

    Used to log a fractional epoch (docs consumed / docs per epoch). Counts the *raw* docs
    (fractional repeats as expectation); a few are dropped at L3 (truncation that removes
    all targets), so the fraction is a hair optimistic, which is fine for a progress axis.
    Returns 0 if it can't be computed."""
    from data_mix import count_mix_docs

    try:
        return count_mix_docs(dataset_paths, mix_entries)
    except Exception as e:  # noqa: BLE001 -- a metric must never break training
        logger.warning(f"count_dataset_docs failed ({e}); fractional epoch disabled")
        return 0


def iter_packed_bins(example_iter: Iterator[Example], micro_len: int, pad_id: int,
                     max_segs: Optional[int], buf_size: int = SHUFFLE_BUF,
                     shuffle_seed: Optional[int] = None):
    """Buffer examples then greedy length-pack them into fixed `micro_len` rows (on CPU).

    Greedy first-fit-decreasing is not streamable (it sorts), so we pack in windows of
    `buf_size` examples: fill the buffer, pack it into bins, yield them, repeat. Each bin
    materializes to padded [1, micro_len] CPU tensors (+ fixed-shape varlen metadata when
    `max_segs` is set for torch.compile stability); the main loop moves them to GPU. The
    window is kept small so the first bin (and thus the first optimizer step) is ready
    quickly -- with long L2 docs (~12-18k median) only ~2-3 fit per 32k row, so even a
    256-example window packs at ~98%."""
    import random

    buf: list[Example] = []
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None

    def flush():
        if rng is not None:
            rng.shuffle(buf)
        for b in greedy_pack(buf, micro_len):
            yield pack_to_tensors(b, micro_len, pad_id, "cpu", max_segs)
        buf.clear()

    for ex in example_iter:
        buf.append(ex)
        if len(buf) >= buf_size:
            yield from flush()
    if buf:
        yield from flush()


def threaded_prefetch(gen, max_ahead: int = PREFETCH_BINS):
    """Run a (CPU-bound) generator in a background thread, yielding through a bounded queue.

    The L3 render+tokenize+pack pipeline is CPU work; without overlap the GPU starves while
    each window is built (and idles entirely during the first window). The HF fast tokenizer
    releases the GIL during its Rust encode and CUDA kernels release it during compute, so a
    single producer thread genuinely overlaps data prep with the fwd/bwd of earlier bins.
    The queue caps how far ahead we run (bounded memory)."""
    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=max_ahead)
    _DONE = object()
    err: list[BaseException] = []

    def worker():
        try:
            for item in gen:
                q.put(item)
        except BaseException as e:  # noqa: BLE001 -- re-raised on the consumer side
            err.append(e)
        finally:
            q.put(_DONE)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is _DONE:
            break
        yield item
    if err:
        raise err[0]


def _to_device(batch: dict, device: str) -> dict:
    """Move a CPU-materialized packed bin to the compute device (tensors only)."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _next_step_bins(bin_iter, accum: int, world: int, device: str, stop_now: bool = False):
    """Pull `accum` bins for one optimizer step; return None if data is out OR time is up.

    All ranks must run the same number of optimizer steps or collectives deadlock, so we
    all-reduce a MIN over a per-rank "keep going?" flag (0 if this rank is out of data or its
    wall-clock budget is exhausted) and stop together. Because every rank checks the clock at
    ~the same wall time, the time-stop fires on all ranks within the same step."""
    bins = []
    ok = 0 if stop_now else 1
    if ok:
        for _ in range(accum):
            b = next(bin_iter, None)
            if b is None:
                ok = 0
                break
            bins.append(b)
    if world > 1:
        t = torch.tensor([ok], device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        ok = int(t.item())
    return bins if ok else None


def _skip_packed_bins(bin_iter, n_bins: int, rank: int) -> int:
    """Deterministically consume already-trained packed bins after resume.

    This is intentionally a reconstruction skip rather than a fragile cursor checkpoint: the
    iterator is deterministic for (epoch, rank, seed), so skipping the per-rank bin count lands
    on the same next batch as an uninterrupted run. It costs CPU time on resume but avoids
    replaying data after Slurm requeue / FMI continuation.
    """
    skipped = 0
    for _ in range(n_bins):
        if next(bin_iter, None) is None:
            break
        skipped += 1
    logger.info(f"rank {rank}: skipped {skipped}/{n_bins} packed bins for resume")
    return skipped


# ---- LR schedule ----
def lr_lambda(step: int, warmup: int, total: int, decay: str = "cosine",
              min_ratio: float = 0.0) -> float:
    """Multiplier on the base LR: linear warmup, then decay to `min_ratio` * peak by `total`.

    decay="cosine": half-cosine peak->min_ratio. decay="linear": straight line peak->min_ratio.
    After `total` the multiplier holds at `min_ratio` (the floor)."""
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    if decay == "linear":
        factor = 1.0 - progress
    else:  # cosine
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * factor


# ---- checkpoint ----
def _model_full_state_dict(model, world: int):
    """Gather a consolidated (unsharded, CPU) state dict on rank 0 for HF save_pretrained."""
    if world <= 1:
        return model.state_dict()
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    return get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def save_consolidated(model, tokenizer, output_path: str, world: int, rank: int) -> None:
    """Write an HF-loadable checkpoint (final deliverable). Gathers the sharded params."""
    sd = _model_full_state_dict(model, world)
    if is_main(rank):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        # Downcast the deliverable to bf16 (the serving/inference dtype, half the size). When
        # training with an fp32 master this halves the consolidated checkpoint; the fp32 master
        # itself is preserved in the DCP `_resume` checkpoint for exact continuation.
        sd = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in sd.items()}
        model.config.dtype = torch.bfloat16  # transformers v5 field (torch_dtype is the deprecated alias)
        # save_pretrained on the underlying HF module; pass the gathered state dict so we
        # don't try to re-shard. With FSDP2 the model IS the HF module (in-place wrapped).
        to_save = model
        to_save.save_pretrained(output_path, state_dict=sd, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        logger.info(f"saved consolidated HF checkpoint (bf16) -> {output_path}")
    if world > 1:
        torch.distributed.barrier()


def save_resume(model, optimizer, scheduler, step: int, epoch: int, bins_consumed_epoch: int,
                ckpt_dir: str, world: int, rank: int, mix_hash: Optional[str] = None) -> None:
    """Sharded checkpoint (model+optim) via DCP + scalar meta, for exact resume."""
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    # Release cached-but-unused GPU memory back to the driver before the DCP save. dcp.save runs
    # an NCCL reduce_scatter/scatter_object for its save plan; with the caching allocator holding
    # most of HBM (the ~130 GB fp32-32B model-load transient stays reserved even after sharding),
    # NCCL's fresh buffer cudaMalloc fails -> "NCCL WARN Cuda failure 2 'out of memory'" *during
    # the checkpoint*, not during training. empty_cache frees the headroom NCCL needs. (Observed:
    # 76852 trained fine 50+ steps at micro_len 65536, then OOM'd at the first 30-min save.)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if world > 1:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        msd, osd = get_state_dict(model, optimizer)
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=ckpt_dir)
    else:
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict()},
                   os.path.join(ckpt_dir, "state.pt"))
    if is_main(rank):
        meta = {
            "schema_version": 2,
            "step": step,
            "epoch": epoch,
            "bins_consumed_epoch": bins_consumed_epoch,
            "scheduler": scheduler.state_dict(),
            "mix_hash": mix_hash,
        }
        meta_tmp = os.path.join(ckpt_dir, "meta.json.tmp")
        meta_path = os.path.join(ckpt_dir, "meta.json")
        with open(meta_tmp, "w") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, meta_path)
    if world > 1:
        torch.distributed.barrier()
    logger.info(f"saved resume checkpoint (step {step}) -> {ckpt_dir}")


def load_resume(model, optimizer, scheduler, ckpt_dir: str, world: int,
                mix_hash: Optional[str] = None) -> dict:
    """Restore model+optim from a DCP/`state.pt` checkpoint. Returns the meta dict."""
    meta_path = os.path.join(ckpt_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no resume meta at {ckpt_dir}")
    with open(meta_path) as f:
        meta = json.load(f)
    # The deterministic bin-skip resume replays the data stream from the mix; continuing
    # with a different mix would silently re-train/skip the wrong rows -> fail loud.
    if meta.get("mix_hash") is not None and mix_hash is not None and meta["mix_hash"] != mix_hash:
        raise RuntimeError(
            f"resume mix mismatch: checkpoint was written with mix_hash={meta['mix_hash']} "
            f"but this run resolves to {mix_hash}; the data mix must not change across "
            "continuations of one run")
    if world > 1:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        msd, osd = get_state_dict(model, optimizer)
        dcp.load({"model": msd, "optim": osd}, checkpoint_id=ckpt_dir)
        set_state_dict(model, optimizer, model_state_dict=msd, optim_state_dict=osd)
    else:
        sd = torch.load(os.path.join(ckpt_dir, "state.pt"), map_location="cpu")
        model.load_state_dict(sd["model"])
        optimizer.load_state_dict(sd["optim"])
    if "scheduler" in meta:
        scheduler.load_state_dict(meta["scheduler"])
    elif "sched_step" in meta:
        # Backward compatibility with schema v1 checkpoints.
        # v1 stored a submission-local sched_step, which becomes wrong after the second
        # continuation. The absolute optimizer step is the best available reconstruction.
        for _ in range(int(meta["step"])):
            scheduler.step()
    else:
        logger.warning("resume checkpoint has no scheduler state; continuing with fresh scheduler")
    logger.info(f"resumed from {ckpt_dir}: step={meta['step']} epoch={meta['epoch']} "
                f"bins_consumed_epoch={meta.get('bins_consumed_epoch', 0)}")
    return meta


# ---- orchestration ----
def train(
    model_path: str,
    dataset_path: str,
    output_path: str,
    logdir: str,
    *,
    data_mix: Optional[str] = None,    # mix JSON path (None -> every partition x1)
    learning_rate: float = 1e-5,
    sink_lr: Optional[float] = None,   # None -> same as body LR (no special sink treatment)
    master_dtype: str = "auto",        # auto -> fp32 master on FSDP (multi-GPU), bf16 single-GPU
    num_train_epochs: int = 1,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    micro_len: int = 65536,
    max_len: int = 12288,
    max_steps: Optional[int] = None,
    max_hours: float = 24.0,
    warmup_ratio: float = 0.03,
    warmup_steps: int = 100,
    lr_decay: str = "cosine",
    min_lr_ratio: float = 0.0,
    grad_clip: float = 1.0,
    grad_ckpt: bool = True,
    compile_layers: bool = False,
    save_steps: int = 0,
    save_minutes: float = 60.0,
    log_every: int = 10,
    resume: bool = False,
    seed: int = 0,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_mode: str = "online",
    no_save: bool = False,
    distill_teacher: Optional[str] = None,   # frozen teacher checkpoint -> final-hidden MSE distill
    distill_weight: float = 1.0,             # lambda on the (mean-over-dim) hidden MSE term
) -> None:
    rank, world, local = setup_distributed()
    setup_logging(logdir, rank)
    torch.manual_seed(seed)
    device = f"cuda:{local}"
    if max_len > micro_len:
        raise ValueError(f"max_len {max_len} (per-doc truncation) must be <= micro_len {micro_len} (pack row)")
    time_budget = max_hours * 3600.0 if max_hours and max_hours > 0 else None
    # packed bins consumed per optimizer step = per-rank rows/microstep x accum microsteps.
    # Both default to 1 (the large packed row already gives a big effective batch).
    bins_per_step = per_device_batch_size * gradient_accumulation_steps

    logger.info(f"stage1 SFT | rank={rank}/{world} local={local} | model={model_path}")
    logger.info(f"lr={learning_rate} epochs={num_train_epochs} pdbs={per_device_batch_size} "
                f"accum={gradient_accumulation_steps} bins/step={bins_per_step} "
                f"micro_len={micro_len} max_len(trunc)={max_len} max_steps={max_steps} "
                f"max_hours={max_hours} save_minutes={save_minutes} save_steps={save_steps} "
                f"grad_ckpt={grad_ckpt} compile={compile_layers}")

    # Resolve the data source up front (fail loud on a bad config BEFORE the model load).
    # Two modes: a pre-packed L4 directory (preferred -- offline render/shuffle/pack, see
    # build_l4.py) or the legacy streaming path over L2 roots with a weighted mix.
    from data_mix import describe, load_mix, mix_fingerprint
    dataset_paths = [Path(p) for p in dataset_path.split(",") if p]
    l4 = l4_meta(dataset_paths)
    if l4 is not None:
        if data_mix:
            raise ValueError("--data-mix is baked into an L4 dataset at build time; "
                             "drop the flag (L4 meta records the mix fingerprint)")
        if l4["micro_len"] != micro_len:
            raise ValueError(f"L4 was packed at micro_len {l4['micro_len']} "
                             f"but --micro-len is {micro_len}")
        mix_entries = []
        mix_hash = f"l4:{l4['mix_fingerprint']}:{l4['n_bins']}"
        if is_main(rank):
            logger.info(f"L4 dataset: {dataset_paths[0]} | {l4['n_bins']:,} bins, "
                        f"{l4['n_docs']:,} docs, fill {l4['fill']:.1%} | mix "
                        f"{l4['mix_fingerprint']} | hash {mix_hash}")
    else:
        mix_entries = load_mix(data_mix) if data_mix else []
        mix_hash = mix_fingerprint(dataset_paths, mix_entries)
        if is_main(rank):
            logger.info(f"data mix: {data_mix or '(none, all x1)'} | hash {mix_hash}\n"
                        + describe(dataset_paths, mix_entries))

    wandb_run = setup_wandb(
        wandb_project, wandb_run_name, wandb_mode, logdir, rank,
        config=dict(model_path=model_path, world=world, lr=learning_rate, sink_lr=sink_lr,
                    master_dtype=master_dtype, epochs=num_train_epochs,
                    per_device_batch_size=per_device_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    micro_len=micro_len, max_len=max_len, max_steps=max_steps,
                    max_hours=max_hours, grad_ckpt=grad_ckpt, compile=compile_layers, seed=seed,
                    data_mix=data_mix, mix_hash=mix_hash))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id

    # Resolve master weight dtype. fp32 master + bf16 compute only works through FSDP2's
    # MixedPrecisionPolicy (multi-GPU); the single-GPU bnb path has no such cast and FA3 wants
    # bf16 inputs, so single-GPU always uses bf16. "auto" -> fp32 on FSDP, bf16 single-GPU.
    if master_dtype == "auto":
        use_fp32_master = world > 1
    elif master_dtype == "fp32":
        use_fp32_master = world > 1
        if world <= 1:
            logger.warning("--master-dtype fp32 is only supported on the FSDP (multi-GPU) path; "
                           "falling back to bf16 on single GPU")
    else:  # "bf16"
        use_fp32_master = False
    load_dtype = torch.float32 if use_fp32_master else torch.bfloat16
    logger.info(f"master weight dtype = {'fp32 (master) + bf16 compute' if use_fp32_master else 'bf16'}")

    # ce-only Liger pairs with compile (compile fuses rope/norm/swiglu); otherwise full Liger.
    liger_mode = "ce-only" if compile_layers else True
    model = build_model(model_path, liger=liger_mode, master_dtype=load_dtype)
    if grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False})
    model = setup_parallelism(model, world, local)
    model.config.use_cache = False
    model.train()

    # ---- optional MHA->GQA distillation: frozen teacher, final-hidden MSE ----
    # Student (GQA) and teacher (e.g. the original MHA) share embed/lm_head/FFN/norm; only
    # KV heads differ. Matching the final pre-lm_head hidden state pins the student to the
    # teacher's representation (=> its logits, since lm_head is shared) with a dense signal
    # and no [S, vocab] logit materialization. Teacher runs replicated per rank (not FSDP),
    # forward-only, logits_to_keep=1 to skip ITS logits. (2026-06-13: 32->8 KV heads are
    # ~orthogonal, so pooling can't give a good init; the gap closes via uptrain, and this
    # teacher signal closes it further than LM CE alone -- see docs/gqa_conversion.md.)
    teacher = None
    _hid: dict = {}
    if distill_teacher:
        def _grab(tag):
            def hook(_m, _i, out):
                _hid[tag] = out
            return hook
        teacher = build_model(distill_teacher, liger=False, master_dtype=torch.bfloat16)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.model.norm.register_forward_hook(_grab("t"))
        model.model.norm.register_forward_hook(_grab("s"))
        logger.info(f"distill: teacher={distill_teacher} final-hidden MSE weight={distill_weight}")

    if compile_layers:
        for i in range(len(model.model.layers)):
            model.model.layers[i] = torch.compile(model.model.layers[i], dynamic=False)

    # When compiling, emit FIXED-shape varlen metadata (else varying #docs busts Dynamo's
    # recompile cache -> eager fallback). Bound segments per bin; greedy_pack stays under it
    # for realistic doc lengths. Without compile, variable shapes are fine (eager varlen).
    max_segs = (micro_len // 256 + 2) if compile_layers else None

    # Sinks train at the body LR by default (no special treatment): with SINK_INIT=0.0 the
    # sink logits sit near zero where bf16 resolves fine, get a real (if small) gradient, and
    # move organically as they start mattering -- the init-0 probe shows Δ≈9e-5/40 steps at
    # 1e-5, i.e. nonzero and self-accelerating, vs exactly 0 for the old -10 dead-zone. A
    # separate, larger `sink_lr` is available as an OPTIONAL experimental knob (--sink-lr) but
    # is OFF by default: cranking one param group to 100-1000x the body LR is destabilising
    # and amplifies a weak/noisy signal rather than learning. If you do want faster sink
    # training, prefer fp32 sinks + a mild bump over a big bf16 LR. The LR scheduler scales
    # all groups by the same warmup/cosine factor.
    eff_sink_lr = sink_lr if sink_lr is not None else learning_rate
    sink_params = [p for n, p in model.named_parameters() if n.endswith("self_attn.sinks")]
    body_params = [p for n, p in model.named_parameters() if not n.endswith("self_attn.sinks")]
    if eff_sink_lr == learning_rate:
        param_groups = model.parameters()  # uniform LR -> single group (no special treatment)
    else:
        param_groups = [{"params": body_params, "lr": learning_rate},
                        {"params": sink_params, "lr": eff_sink_lr}]
    logger.info(f"optimizer: body lr {learning_rate} | {len(sink_params)} sink params lr {eff_sink_lr}"
                + ("" if eff_sink_lr == learning_rate else " (separate group)"))
    # optimizer: fused AdamW under FSDP (state sharded); PagedAdamW8bit single-GPU (state on CPU).
    if world > 1:
        opt = torch.optim.AdamW(param_groups, lr=learning_rate, fused=True,
                                betas=(0.9, 0.95), weight_decay=0.0)
    else:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(param_groups, lr=learning_rate,
                                       betas=(0.9, 0.95), weight_decay=0.0)

    # LR schedule. With --max-steps -> warmup + cosine decay to 0 over that horizon. Without
    # it (the default time-/epoch-bounded run, where the step count isn't known ahead and the
    # run may continue across FMI submissions via --resume) -> warmup then HOLD constant; a
    # final LR anneal, if wanted, is a separate bounded submission. lr_lambda holds at 1.0
    # past `warmup` when warmup==horizon-style large, so we give it a huge horizon for the
    # constant case (cosine arg saturates to min(1.0, progress) -> never reaches the decay).
    if max_steps:
        horizon, warmup = max_steps, max(1, int(warmup_ratio * max_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: lr_lambda(s, warmup, horizon, decay=lr_decay, min_ratio=min_lr_ratio))
        if is_main(rank):
            logger.info(f"LR: warmup {warmup} steps then {lr_decay} decay to "
                        f"{min_lr_ratio:g}*peak ({min_lr_ratio * learning_rate:.2e}) over {horizon} steps")
    else:
        warmup = warmup_steps
        # warmup ramp then constant 1.0 (no decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, warmup)))
        if is_main(rank):
            logger.info(f"LR: warmup {warmup} steps then constant {learning_rate} "
                        f"(time/epoch-bounded run; pass --max-steps for decay).")

    start_step = start_epoch = start_bins_consumed_epoch = 0
    resume_dir = os.path.join(output_path, "_resume")
    if resume and os.path.exists(os.path.join(resume_dir, "meta.json")):
        meta = load_resume(model, opt, sched, resume_dir, world, mix_hash=mix_hash)
        start_step = meta["step"]
        start_epoch = meta["epoch"]
        start_bins_consumed_epoch = int(meta.get("bins_consumed_epoch", start_step * bins_per_step))

    # ---- step loop ----
    # Reset the CUDA peak so the logged "peak" reflects TRAINING memory, not the one-time
    # model-load transient. setup_parallelism moves the full unsharded model to one GPU before
    # FSDP shards it (fp32 32B = ~130 GB!); that spike masks the real per-step peak otherwise.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step = start_step
    t_win = time.perf_counter()
    t_start = time.perf_counter()     # wall-clock anchor for the time budget
    t_last_save = t_start             # for time-based periodic checkpointing
    tok_win = 0
    tokens_seen = 0                   # cumulative global tokens this submission (wandb x-axis)
    # docs/epoch for a fractional-epoch metric (rank 0 only; counted once from parquet metadata)
    docs_per_epoch = (l4["n_docs"] if l4 is not None
                      else count_dataset_docs(dataset_paths, mix_entries)) if is_main(rank) else 0
    done = False
    stop_reason = None
    for epoch in range(start_epoch, num_train_epochs):
        if l4 is not None:
            cpu_bins = iter_l4_bins(dataset_paths[0], l4, rank, world, epoch, max_segs)
        else:
            ex_iter = iter_examples(dataset_paths, mix_entries, tokenizer, max_len, rank, world,
                                    epoch=epoch)
            pack_seed = seed + 1009 * epoch + 1_000_003 * rank
            cpu_bins = iter_packed_bins(ex_iter, micro_len, pad_id, max_segs,
                                        shuffle_seed=pack_seed)
        bin_iter = threaded_prefetch(cpu_bins)  # overlap data IO/CPU with GPU compute
        bins_consumed_epoch = 0       # per-rank packed bins consumed in this epoch
        if epoch == start_epoch and start_bins_consumed_epoch:
            skipped = _skip_packed_bins(bin_iter, start_bins_consumed_epoch, rank)
            if skipped != start_bins_consumed_epoch:
                stop_reason = "epoch-end"
                break
            bins_consumed_epoch = start_bins_consumed_epoch
        docs_epoch = 0                # docs consumed so far this epoch (rank0 local x world ~ global)
        while True:
            time_up = time_budget is not None and (time.perf_counter() - t_start) >= time_budget
            # rank-synced stop: out-of-data on any rank OR time budget exhausted -> all stop
            bins = _next_step_bins(bin_iter, bins_per_step, world, device,
                                   stop_now=time_up)
            if bins is None:
                stop_reason = "time-budget" if time_up else "epoch-end"
                break
            opt.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            step_loss_tok = 0
            local_target_tokens = sum(int((b["labels"] != IGNORE).sum()) for b in bins)
            if local_target_tokens <= 0:
                raise RuntimeError("packed optimizer step has no supervised target tokens")
            global_target_tokens = local_target_tokens
            if world > 1:
                tt = torch.tensor([local_target_tokens], device=device, dtype=torch.float32)
                torch.distributed.all_reduce(tt, op=torch.distributed.ReduceOp.SUM)
                global_target_tokens = int(tt.item())
            step_distill = torch.zeros((), device=device)
            for b in bins:
                n_target = int((b["labels"] != IGNORE).sum())
                b = _to_device(b, device)
                if teacher is not None:
                    bt = {k: v for k, v in b.items() if k != "labels"}
                    with torch.no_grad():
                        teacher(**bt, logits_to_keep=1)
                out = model(**b)
                loss = out.loss
                if teacher is not None:
                    # MSE over non-pad positions, meaned over the hidden dim (=> ~CE scale,
                    # lambda tunes it). Match real tokens only (pad rows carry no signal).
                    nonpad = (b["input_ids"] != pad_id)[0]                    # [S]
                    sh = _hid["s"][0].float()[nonpad]                         # [N, H]
                    th = _hid["t"][0].float()[nonpad]
                    dloss = ((sh - th) ** 2).mean(dim=-1).mean()
                    loss = loss + distill_weight * dloss
                    step_distill += dloss.detach() * n_target
                # `out.loss` is already mean CE over this bin's non-IGNORE labels. Weight
                # each bin by supervised-token count so gradient accumulation optimizes the
                # global token-level SFT objective instead of giving every packed row equal
                # weight. FSDP/DDP averages grads across ranks, so multiply by `world`.
                scale = (n_target * world) / max(1, global_target_tokens)
                (loss * scale).backward()
                step_loss += out.loss.detach() * n_target
                step_loss_tok += n_target
            gnorm = None
            if grad_clip and grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # FSDP2 returns the (global) total norm as a replicated DTensor; pull a float.
                gnorm = gn.full_tensor().item() if hasattr(gn, "full_tensor") else float(gn)
            opt.step()
            sched.step()
            step += 1
            tok_win += micro_len * len(bins)
            tokens_seen += micro_len * len(bins) * world
            bins_consumed_epoch += len(bins)
            docs_epoch += sum(int(bb.get("n_docs", 0)) for bb in bins) * world  # ~global (per-rank x world)

            if step % log_every == 0:
                torch.cuda.synchronize()
                dt = time.perf_counter() - t_win
                tps = tok_win / dt if dt > 0 else 0
                avg_loss = (step_loss / max(1, local_target_tokens)).item()
                avg_distill = (step_distill / max(1, local_target_tokens)).item()
                if is_main(rank):
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    el = (time.perf_counter() - t_start) / 3600.0
                    cur_lr = sched.get_last_lr()[0]
                    # fractional epoch: integer epoch + fraction of this epoch's docs consumed
                    frac_epoch = epoch + (min(0.9999, docs_epoch / docs_per_epoch) if docs_per_epoch else 0.0)
                    gn_str = f"{gnorm:.3f}" if gnorm is not None else "n/a"
                    dstr = f" | distill {avg_distill:.4f}" if teacher is not None else ""
                    logger.info(f"step {step} | ep {frac_epoch:.4f} | loss {avg_loss:.4f}{dstr} | lr {cur_lr:.2e} "
                                f"| gnorm {gn_str} | {tps * world:,.0f} tok/s global ({tps:,.0f}/gpu) "
                                f"| loss_tok/step {step_loss_tok} | peak {mem:.1f}GB | {el:.2f}h")
                    if wandb_run is not None:
                        logd = {
                            "train/loss": avg_loss, "train/lr": cur_lr,
                            "perf/tok_s_global": tps * world, "perf/tok_s_gpu": tps,
                            "perf/peak_gb": mem, "train/loss_tok_per_step": step_loss_tok,
                            "time/elapsed_h": el, "train/epoch": frac_epoch,
                            "train/tokens": tokens_seen,
                        }
                        if teacher is not None:
                            logd["train/distill_mse"] = avg_distill
                        if gnorm is not None:
                            logd["train/grad_norm"] = gnorm
                        wandb_run.log(logd, step=step)
                t_win = time.perf_counter()
                tok_win = 0

            # periodic checkpoint: by step count and/or wall-clock minutes (crash recovery +
            # cross-submission continuation). Both ranks hit the same step so the collective
            # save stays in lockstep.
            by_step = save_steps and step % save_steps == 0
            by_time = save_minutes and (time.perf_counter() - t_last_save) >= save_minutes * 60.0
            # RANK-SYNC the wall-clock trigger. Each rank reads its OWN clock; when the
            # save-interval boundary lands inside the few-ms inter-rank step skew, part of
            # the world enters dcp.save's collectives while the rest enters the next step's
            # forward all-gather -> mismatched collectives -> permanent spin (root cause of
            # jobs 79388 [watchdog ALLREDUCE NumelIn=1 after 600s] and 79786 [40-min stall,
            # ranks in R-state NCCL busy-wait, empty _resume]). MAX-reduce makes the
            # decision unanimous; one extra 4-byte allreduce per step is noise.
            if world > 1 and save_minutes:
                t = torch.tensor([1.0 if by_time else 0.0], device=device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
                by_time = t.item() > 0
            if by_step or by_time:
                save_resume(model, opt, sched, step, epoch, bins_consumed_epoch,
                            resume_dir, world, rank, mix_hash=mix_hash)
                t_last_save = time.perf_counter()

            if max_steps and step >= max_steps:
                done, stop_reason = True, "max-steps"
                break
        if stop_reason == "epoch-end" and not done:
            start_bins_consumed_epoch = 0
        if done or stop_reason == "time-budget":
            break

    el = (time.perf_counter() - t_start) / 3600.0
    if no_save:
        logger.info(f"training stopped ({stop_reason}) at step {step} after {el:.2f}h; --no-save (benchmark), skipping checkpoints")
    else:
        logger.info(f"training stopped ({stop_reason}) at step {step} after {el:.2f}h; writing final checkpoint")
        # final: DCP resume (model+optim, for the next submission) + consolidated HF weights (deliverable)
        save_epoch = epoch
        save_bins = locals().get("bins_consumed_epoch", 0)
        if stop_reason == "epoch-end":
            save_epoch = min(epoch + 1, num_train_epochs)
            save_bins = 0
        save_resume(model, opt, sched, step, save_epoch, save_bins, resume_dir, world, rank,
                    mix_hash=mix_hash)
        save_consolidated(model, tokenizer, output_path, world, rank)
    if wandb_run is not None:
        wandb_run.summary["final_step"] = step
        wandb_run.summary["stop_reason"] = stop_reason
        wandb_run.finish()
    if world > 1:
        torch.distributed.destroy_process_group()


def main() -> None:
    # Per-rank, node-local Triton JIT cache. Liger's RoPE/etc. kernels compile lazily on the
    # first forward; with many ranks sharing one cache dir on a distributed FS (WekaFS ~/.triton)
    # the concurrent cold compile races on the .cubin atomic-rename/visibility and dies with
    # `FileNotFoundError: .../_triton_rope.cubin`. A unique node-local dir per rank removes the
    # race entirely (tiny one-time recompile per rank). Set before any kernel launch. Observed
    # on the 32B 4-node run (32 ranks cold-compiling at once); 7B didn't hit it (cache pre-warmed).
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        f"/tmp/triton_{os.environ.get('SLURM_JOB_ID', 'x')}_{os.environ.get('RANK', '0')}")
    p = argparse.ArgumentParser(description="Olmo3Sink stage-1 SFT (FMI train variant)")
    # FMI-required
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset_path", required=True,
                   help="L2 root (hive: dataset=*/domain=*); comma-separate multiple roots "
                        "(mix entries match roots by basename)")
    p.add_argument("--output_path", required=True)
    p.add_argument("--logdir", required=True)
    p.add_argument("--data-mix", dest="data_mix", default=None,
                   help="mix JSON assigning per-partition repeat weights across the roots "
                        "(see data_mix.py; default: every partition x1)")
    # FMI-optional (standard names)
    p.add_argument("--num_gpus", type=int, default=None, help="informational; parallelism is env-driven (torchrun)")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--sink-lr", dest="sink_lr", type=float, default=None,
                   help="OPTIONAL separate LR for the sink logits (default: same as body LR; a big bump is destabilising)")
    p.add_argument("--master-dtype", dest="master_dtype", choices=["auto", "bf16", "fp32"], default="auto",
                   help="optimizer master-weight dtype: auto=fp32 master on multi-GPU/bf16 single-GPU (fp32 fixes bf16 stale-weights, ~same speed)")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_batch_size", type=int, default=1,
                   help="packed rows/rank/microstep; bins/optimizer-step = this x --gradient_accumulation_steps")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="grad-accum microbatches/step; the large packed row already gives a big batch")
    # ours
    p.add_argument("--micro-len", dest="micro_len", type=int, default=65536,
                   help="tokens per packed row (one bin); packs many max_len-capped docs. 65536 fits H100/H200; raise on big-memory cards")
    p.add_argument("--max-len", dest="max_len", type=int, default=12288,
                   help="per-doc truncation length (right-truncate longer docs; stage-1 is short-context)")
    p.add_argument("--max-steps", dest="max_steps", type=int, default=None,
                   help="stop after this many optimizer steps; also the cosine horizon (default: run to --max-hours / epoch end)")
    p.add_argument("--max-hours", dest="max_hours", type=float, default=24.0,
                   help="wall-clock budget: stop + checkpoint when reached (default 24h; 0 disables)")
    p.add_argument("--warmup-ratio", dest="warmup_ratio", type=float, default=0.03,
                   help="warmup fraction of --max-steps (cosine path only)")
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=100,
                   help="warmup steps for the constant-LR (no --max-steps) path")
    p.add_argument("--lr-decay", dest="lr_decay", choices=["cosine", "linear"], default="cosine",
                   help="LR decay shape over --max-steps (cosine or linear); needs --max-steps")
    p.add_argument("--min-lr-ratio", dest="min_lr_ratio", type=float, default=0.0,
                   help="decay floor as a fraction of peak LR (e.g. 0.1 = decay to 0.1*peak, then hold)")
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--grad-ckpt", dest="grad_ckpt", action=argparse.BooleanOptionalAction, default=True,
                   help="activation checkpointing (needed for long packed rows)")
    p.add_argument("--compile", dest="compile_layers", action="store_true",
                   help="torch.compile decoder layers (+ ce-only Liger + fixed-shape varlen)")
    p.add_argument("--save-steps", dest="save_steps", type=int, default=0,
                   help="periodic DCP checkpoint every N steps (0 = off; use --save-minutes)")
    p.add_argument("--save-minutes", dest="save_minutes", type=float, default=60.0,
                   help="periodic DCP checkpoint every M wall-clock minutes (0 = off)")
    p.add_argument("--log-every", dest="log_every", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="resume from <output_path>/_resume if present")
    p.add_argument("--seed", type=int, default=0)
    # wandb (rank-0 logging; off unless --wandb-project given)
    p.add_argument("--wandb-project", dest="wandb_project", default=os.environ.get("WANDB_PROJECT"),
                   help="enable wandb logging to this project (rank 0 only; default off / $WANDB_PROJECT)")
    p.add_argument("--wandb-run-name", dest="wandb_run_name", default=os.environ.get("WANDB_RUN_NAME"),
                   help="wandb run display name (default: wandb auto-name / $WANDB_RUN_NAME)")
    p.add_argument("--wandb-mode", dest="wandb_mode",
                   choices=["online", "offline", "disabled"],
                   default=os.environ.get("WANDB_MODE", "online"),
                   help="online (needs egress+key), offline (sync later), or disabled")
    p.add_argument("--no-save", dest="no_save", action="store_true",
                   help="skip the final DCP+HF checkpoint writes (for throughput benchmarking)")
    p.add_argument("--distill-teacher", dest="distill_teacher", default=None,
                   help="frozen teacher checkpoint; adds final-hidden MSE distillation (e.g. MHA->GQA)")
    p.add_argument("--distill-weight", dest="distill_weight", type=float, default=1.0,
                   help="lambda on the hidden-MSE distill term")
    a = p.parse_args()
    train(
        a.model_path, a.dataset_path, a.output_path, a.logdir,
        data_mix=a.data_mix,
        learning_rate=a.learning_rate, sink_lr=a.sink_lr, master_dtype=a.master_dtype,
        num_train_epochs=a.num_train_epochs, per_device_batch_size=a.per_device_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps, micro_len=a.micro_len,
        max_len=a.max_len, max_steps=a.max_steps, max_hours=a.max_hours,
        warmup_ratio=a.warmup_ratio, warmup_steps=a.warmup_steps,
        lr_decay=a.lr_decay, min_lr_ratio=a.min_lr_ratio,
        grad_clip=a.grad_clip, grad_ckpt=a.grad_ckpt, compile_layers=a.compile_layers,
        save_steps=a.save_steps, save_minutes=a.save_minutes, log_every=a.log_every,
        resume=a.resume, seed=a.seed,
        wandb_project=a.wandb_project, wandb_run_name=a.wandb_run_name, wandb_mode=a.wandb_mode,
        no_save=a.no_save,
        distill_teacher=a.distill_teacher, distill_weight=a.distill_weight,
    )


if __name__ == "__main__":
    main()

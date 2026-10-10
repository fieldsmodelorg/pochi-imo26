# Copyright 2026 proof-pilot. Apache-2.0.
"""trainer-as-service — rank-0 HTTP ingress + command-loop across all ranks (V15/V17).

HTTP is a single-endpoint request/response, but the HSDP trainer is N ranks that must run collectives in
lockstep. The solution (PLAN §6): **rank-0 is the HTTP ingress + all ranks run a command-loop** — each HTTP
call is translated into "rank-0 broadcasts a command to all ranks, everyone runs the same collective
together". A fixed command set (train_step/save/stop), not arbitrary RPC.

Topology:
  torchrun launches N ranks (cross-node = a single cross-node srun):
    rank-0:   [thread] http.server (stdlib, no fastapi dependency)  <- the only external entry point
              [main]   command-loop (holds CUDA + all torch.distributed)
              the two are bridged via a queue + concurrent.futures.Future
    rank>0:   only the command-loop (blocking, waiting for rank-0's broadcast)
Iron rule: all dist/CUDA collectives run only on the main thread (the command-loop).

Two internal channels: command broadcast (gloo broadcast_object_list, small dict), data scatter (gloo
scatter_object_list, each rank's shard = wire-trajs, **no hidden bytes**). The NCCL PG is reserved for
fwd/bwd + all_reduce.

Endpoints: POST /train_step, POST /save, GET /health, POST /stop.
Launch: `torchrun ... -m opd_v2.trainer.service --run-dir <run>`.
"""
from __future__ import annotations

import argparse
import datetime
import http.server
import json
import logging
import os
import queue
import socket
import sys
import threading
from concurrent.futures import Future

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))   # opd_v2/src

from opd_v2.config import OPDConfig  # noqa: E402
from opd_v2.trainer.core import OPDTrainerV2, lpt_assign  # noqa: E402

log = logging.getLogger("opd_v2.trainer.service")


# ---------------------------------------------------------------------------
# gloo object collectives helper
# ---------------------------------------------------------------------------
def _bcast(obj, group):
    box = [obj]
    torch.distributed.broadcast_object_list(box, src=0, group=group)
    return box[0]


def _scatter(shards, group, world: int, rank: int):
    out = [None]
    inp = shards if rank == 0 else None
    torch.distributed.scatter_object_list(out, inp, src=0, group=group)
    return out[0]


# ---------------------------------------------------------------------------
# HTTP ingress (rank-0, side thread)
# ---------------------------------------------------------------------------
class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, inbox: queue.Queue, get_step):
        super().__init__(addr, handler)
        self.inbox = inbox
        self.get_step = get_step


class _Handler(http.server.BaseHTTPRequestHandler):
    _OPS = {"/train_step": "train_step", "/save": "save", "/checkpoint": "checkpoint", "/stop": "stop"}

    def log_message(self, *a):
        pass  # silence (high-frequency train_step)

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "step": self.server.get_step()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        op = self._OPS.get(self.path)
        if op is None:
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n)) if n else {}
        except Exception as e:
            self._send(400, {"error": f"bad json: {e}"})
            return
        fut: Future = Future()
        self.server.inbox.put((op, body, fut))
        try:
            res = fut.result(timeout=7200)        # a train_step's long forward can take a while
        except Exception as e:
            self._send(500, {"error": repr(e)})
            return
        self._send(200, res)


def _start_http(cfg: OPDConfig, inbox: queue.Queue, get_step) -> _Server:
    host = "0.0.0.0"
    port = cfg.trainer.http_port
    srv = _Server((host, port), _Handler, inbox, get_step)
    threading.Thread(target=srv.serve_forever, name="trainer-http", daemon=True).start()
    # endpoint self-registration (V20): write hostname:port to shared FS for the orchestrator to discover
    node = socket.gethostname()
    ep = {"host": node, "port": port, "url": f"http://{node}:{port}"}
    os.makedirs(cfg.run_dir, exist_ok=True)
    tmp = cfg.trainer_endpoint_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ep, f)
    os.replace(tmp, cfg.trainer_endpoint_file)
    log.info("rank-0 HTTP ingress on %s:%d (advertised %s)", host, port, ep["url"])
    return srv


# ---------------------------------------------------------------------------
# command-loop (all ranks, main thread)
# ---------------------------------------------------------------------------
def command_loop_rank0(trainer: OPDTrainerV2, cfg: OPDConfig, gloo, inbox: queue.Queue,
                       world: int):
    cap = cfg.trainer.micro_batch_tokens
    while True:
        op, body, fut = inbox.get()
        try:
            if op == "stop":
                if world > 1:
                    _bcast({"op": "stop"}, gloo)
                fut.set_result({"ok": True, "step": trainer.step})
                break
            if op == "save":
                if world > 1:
                    _bcast({"op": "save", "slot": body.get("slot")}, gloo)
                wi = trainer.save_weights(body.get("slot"))
                fut.set_result(wi)
                continue
            if op == "checkpoint":
                want_hf = bool(body.get("hf", True))
                keep = int(body.get("keep", -1))
                if world > 1:
                    _bcast({"op": "checkpoint", "hf": want_hf, "keep": keep}, gloo)
                info = trainer.save_checkpoint(want_hf=want_hf, keep=keep)
                fut.set_result(info)
                continue
            if op == "train_step":
                want_g4 = bool(body.get("want_g4", False))
                trajs = body.get("trajs", [])
                if world > 1:
                    _bcast({"op": "train_step", "want_g4": want_g4}, gloo)
                    shards, dropped = lpt_assign(trajs, world, cap,
                                                 length_fn=lambda w: len(w["ids"]))
                    my = _scatter(shards, gloo, world, 0)
                else:
                    my, dropped = trajs, []
                m = trainer.train_step(my, want_g4=want_g4)
                m["dropped"] = dropped       # wire-trajs dropped by LPT (orchestrator GCs their handles)
                fut.set_result(m)
                continue
            fut.set_result({"error": f"unknown op {op}"})
        except Exception as e:
            log.exception("command %s failed", op)
            fut.set_result({"error": repr(e)})
            raise                            # collectives may already be desynced -> let the process die, watchdog requeues


def command_loop_worker(trainer: OPDTrainerV2, gloo, world: int, rank: int):
    while True:
        cmd = _bcast(None, gloo)
        op = cmd["op"]
        if op == "stop":
            break
        if op == "save":
            trainer.save_weights(cmd.get("slot"))
        elif op == "checkpoint":
            trainer.save_checkpoint(want_hf=cmd.get("hf", True), keep=cmd.get("keep", -1))
        elif op == "train_step":
            my = _scatter(None, gloo, world, rank)
            trainer.train_step(my, want_g4=False)   # workers don't compute g4 (only rank-0 does)


# ---------------------------------------------------------------------------
# main (torchrun entry)
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("OPD_RUN_DIR", ""))
    a = ap.parse_args()
    if not a.run_dir:
        raise SystemExit("--run-dir (or OPD_RUN_DIR) required")

    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    logging.basicConfig(level=logging.INFO,
                        format=f"%(asctime)s [rank{rank}] %(name)s: %(message)s")
    cfg = OPDConfig.load(a.run_dir)

    torch.cuda.set_device(local)
    gloo = None
    if world > 1:
        backend = "cpu:gloo,cuda:nccl" if cfg.trainer.cpu_offload else "nccl"
        torch.distributed.init_process_group(
            backend, device_id=torch.device(f"cuda:{local}"),
            timeout=datetime.timedelta(minutes=int(os.environ.get("DIST_TIMEOUT_MIN", "60"))))
        gloo = torch.distributed.new_group(backend="gloo",
                                           timeout=datetime.timedelta(hours=2))

    if rank == 0:
        log.info("building OPDTrainerV2 (student=%s world=%d lr=%.2e β=%.2f micro=%d)",
                 cfg.trainer.student_path, world, cfg.trainer.lr, cfg.loss.beta,
                 cfg.trainer.micro_batch_tokens)
    trainer = OPDTrainerV2(cfg, world, local, gloo_group=gloo)
    if rank == 0:
        log.info("OPDTrainerV2 built; FSDP engaged.")

    # resume-on-startup (all-rank collective DCP load; done before HTTP comes up, so the /health.step the
    # orchestrator discovers is already the resumed step). All ranks read the same latest.json -> consistent load decision.
    if cfg.trainer.resume or cfg.trainer.resume_from:
        resumed = trainer.try_resume(cfg.trainer.resume_from)
        if rank == 0:
            if resumed is not None:
                log.info("RESUMED from durable checkpoint: step=%d", resumed)
            else:
                log.info("no resume checkpoint found; starting fresh (step=0)")

    if rank == 0:
        inbox: queue.Queue = queue.Queue()
        _start_http(cfg, inbox, get_step=lambda: trainer.step)
        command_loop_rank0(trainer, cfg, gloo, inbox, world)
    else:
        command_loop_worker(trainer, gloo, world, rank)

    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

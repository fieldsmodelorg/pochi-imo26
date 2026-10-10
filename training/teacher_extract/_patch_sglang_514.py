# Copyright 2026 proof-pilot. Apache-2.0.
"""Patch sglang (0.5.14) files for correct DeepSeek-V4 hidden-state extraction.

Re-anchored from _patch_sglang.py (which targeted 0.5.12.post1). Same four
patches and same intent; anchors are asserted so a changed upstream fails loudly.

What changed for 0.5.14 (see _patch_sglang.py for the original anchors):
  * deepseek_v4.py: a blank line was inserted between
    `hidden_states, pre_hc_head = hidden_states` and `return self.logits_processor(`,
    and logits_processor is now called with `hidden_states_before_norm=pre_hc_head`.
    The hidden-gate anchor is updated to include that blank line. The
    `import os`, load_weights fp8-wo_a dequant, and `_dequant_fp8_wo_a` helper
    anchors are unchanged (the dequant path still exists in 0.5.14, now guarded
    by `if not envs.SGLANG_OPT_FP8_WO_A_GEMM.get():` — we only replace its body).
  * scheduler.py: unchanged anchor (the `if batch.return_logprob:` snapshot block
    now also assigns extend_logprob_start_len_per_req, but our anchor is the
    unchanged prefix so that stays intact).
  * THE OUTPUT-PROCESSOR MIXIN IS GONE. In 0.5.14 `process_batch_result_prefill`
    lives in `managers/scheduler_components/batch_result_processor.py` and the
    finished-prefill hidden append was factored into the method
    `_append_prefill_hidden_states(...)`. We (a) rewrite that helper to slice by
    this forward's extend length (not len(origin_input_ids)) and route through
    `_pp_store_hidden`, (b) pass the per-forward extend length from the
    finished-prefill call site, and (c) add the same append in the being-chunked
    branch (which upstream still omits, so intermediate chunks are dropped).
  * http_server.py: unchanged anchor. Note 0.5.14 has a native `/v1/score`
    route; our bare `/score` route does not collide with it.

Original intent (unchanged):

1. models/deepseek_v4.py — suppress hidden_states_before_norm (pre-hc_head
   [seq,16384], MTP-only) so return_hidden_states yields the post-hc_head
   post-norm [seq,4096] tensor that feeds lm_head. Env-guarded:
   SGLANG_DSV4_HIDDEN_POST_NORM=1.

2. managers/scheduler.py — extend_input_len_per_req is only snapshotted when
   return_logprob; also snapshot when return_hidden_states (output processing
   needs it, and reading req.extend_input_len there is racy under overlap
   scheduling).

3. managers/scheduler_components/batch_result_processor.py —
   process_batch_result_prefill:
   a) the finished-prefill hidden append sliced the batch hidden tensor with
      len(req.origin_input_ids) per req, but the tensor holds only THIS forward's
      extend tokens -> misassignment whenever reqs are packed/chunked (upstream
      #8066 family). Slice with this forward's extend length instead.
   b) the being-chunked branch never appends hidden -> intermediate chunks are
      dropped (only the last chunk was returned). Append there too; pieces
      accumulate in req.hidden_states across chunks.
   c) spool mode (SGLANG_HIDDEN_SPOOL_DIR): the default list+zmq return path is
      12x slower than the forward (measured 23.4k -> 1.7-2.1k tok/s on TP4).
      The emitting rank writes bf16 tensors to disk; responses carry paths.

4. models/deepseek_v4.py — load_weights materializes the whole checkpoint
   (weights = list(weights), non-fp8-wo_a-gemm path) just to find .wo_a.scale
   pairs for the fp8 wo_a dequant. That pins every safetensors mmap (writable
   MAP_PRIVATE = fully commit-charged), so vm.overcommit_memory=2 hosts need
   ckpt_size x tp_size of CommitLimit — V4-Pro TP8 = 806GB x 8 = 6.4TB; load
   dies deterministically at ~1.66TB (shard 13). Replaced with a streaming
   dequant that pairs each .wo_a.weight with its same-shard .wo_a.scale on the
   fly (verified same-shard for both Flash and Pro checkpoints).

Usage: python3 _patch_sglang_514.py <src_dir> <out_dir>
  <src_dir> holds pristine 0.5.14 copies (deepseek_v4.py, scheduler.py,
  batch_result_processor.py, http_server.py); patched copies land in <out_dir>.
"""
import sys


def patch(src: str, anchor: str, replacement: str, expect: int = 1) -> str:
    n = src.count(anchor)
    assert n == expect, f"anchor found {n}x (expected {expect}): {anchor[:80]!r}"
    return src.replace(anchor, replacement)


def patch_dsv4(src: str) -> str:
    if "\nimport os\n" not in src[:2000]:
        src = patch(src, "import logging\nimport time", "import logging\nimport os\nimport time")
    # (1) hidden gate. 0.5.14 inserted a blank line between the unpack and the
    # logits_processor call and now passes hidden_states_before_norm=pre_hc_head.
    src = patch(
        src,
        """        hidden_states, pre_hc_head = hidden_states

        return self.logits_processor(""",
        """        hidden_states, pre_hc_head = hidden_states
        # PATCH(proof-pilot): logits_processor prefers hidden_states_before_norm when
        # returning hidden states; it is only needed as MTP/EAGLE draft input. For
        # distillation extraction we want the post-hc_head post-norm tensor that feeds
        # lm_head. Do NOT set this env with speculative decoding.
        if os.environ.get("SGLANG_DSV4_HIDDEN_POST_NORM", "0") == "1":
            pre_hc_head = None

        return self.logits_processor(""",
    )
    # (4) load_weights: stream the fp8 wo_a dequant instead of materializing the
    # whole checkpoint. list(weights) pins every safetensors mmap (writable
    # MAP_PRIVATE = fully commit-charged) => strict-overcommit hosts need
    # ckpt_size x tp_size of CommitLimit (V4-Pro TP8: 6.4TB) and die mid-load.
    # In 0.5.14 this block is guarded by `if not envs.SGLANG_OPT_FP8_WO_A_GEMM`,
    # which is left intact; only its body is replaced.
    src = patch(
        src,
        """            weights = list(weights)
            exists_wo_a_scale = any(n.endswith(".wo_a.scale") for n, t in weights)
            if exists_wo_a_scale:
                logger.info("Execute dequant fp8 wo_a")
                weights = _dequant_fp8_wo_a(weights)
            else:
                logger.info("Skip dequant fp8 wo_a")""",
        """            # PATCH(proof-pilot): do NOT list(weights) — it pins every ckpt
            # file mapping for the whole load (commit-charged in full under
            # vm.overcommit_memory=2). Stream and pair wo_a weight/scale on
            # the fly instead (pairs are same-shard in all DSV4 checkpoints).
            logger.info("Streaming dequant fp8 wo_a (proof-pilot patch)")
            weights = _dequant_fp8_wo_a_streaming(weights)""",
    )
    src = patch(
        src,
        "def _dequant_fp8_wo_a(",
        '''def _dequant_fp8_wo_a_streaming(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """PATCH(proof-pilot): streaming variant of _dequant_fp8_wo_a.

    Pairs each .wo_a.weight with its .wo_a.scale as they stream by and yields the
    dequantized weight; everything else passes straight through. Checkpoints
    without wo_a scales (bf16 wo_a) fall out of the pending dict unchanged at the
    end. Pending holds at most the wo_a tensors of in-flight shards (a few MB).
    """
    pending: dict = {}
    for name, t in weights:
        if name.endswith(".wo_a.weight") or name.endswith(".wo_a.scale"):
            base, kind = name.rsplit(".", 1)
            d = pending.setdefault(base, {})
            d[kind] = t
            if len(d) == 2:
                del pending[base]
                yield base + ".weight", _dequant_fp8(d["weight"], d["scale"])
        else:
            yield name, t
    for base, d in pending.items():  # unpaired (no-scale checkpoints): pass through
        for kind, t in d.items():
            yield f"{base}.{kind}", t


def _dequant_fp8_wo_a(''',
    )
    return src


def patch_scheduler(src: str) -> str:
    # 0.5.14: this `if` block now also snapshots extend_logprob_start_len_per_req,
    # but our anchor is the unchanged leading portion, so that stays intact and
    # only the condition is widened to include return_hidden_states.
    return patch(
        src,
        """            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]""",
        """            # PATCH(proof-pilot): hidden-state slicing in output processing also
            # needs the per-forward extend lengths (req.extend_input_len is racy there).
            if batch.return_logprob or batch.return_hidden_states:
                batch_result.extend_input_len_per_req = [
                    req.extend_input_len for req in batch.reqs
                ]""",
    )


HELPER = '''logger = logging.getLogger(__name__)


def _pp_store_hidden(req, hs_slice):
    """PATCH(proof-pilot): hidden return path for distillation extraction.

    Upstream converts the hidden slice to nested Python lists (~1e9 PyObjects for
    250k tokens) and ships them through zmq -- measured 12x slower than the forward
    itself. With SGLANG_HIDDEN_SPOOL_DIR set, the emitting rank instead writes the
    bf16 tensor to disk and the response carries only the file path.
    """
    spool = os.environ.get("SGLANG_HIDDEN_SPOOL_DIR")
    if not spool:
        req.hidden_states.append(hs_slice.cpu().clone().tolist())
        return
    # Only the attn-TP emit rank (0) writes the file; every other rank just records the
    # path. 0.5.14 moved this code off the Scheduler (which had self.attn_tp_rank) onto a
    # frozen dataclass, so use the GLOBAL accessor instead of a now-absent self attr
    # (lazy import: managers -> layers.dp_attention, avoid any module import-order issue).
    from sglang.srt.layers.dp_attention import get_attention_tp_rank
    path = os.path.join(spool, f"{req.rid}.{len(req.hidden_states)}.pt")
    if get_attention_tp_rank() == 0:
        os.makedirs(spool, exist_ok=True)
        torch.save(hs_slice.to(torch.bfloat16).cpu().clone(), path)
    req.hidden_states.append(path)'''


def patch_output_processor(src: str) -> str:
    # (0) helper + os import after the import block
    src = patch(src, "import logging\n", "import logging\nimport os\n")
    src = patch(src, "logger = logging.getLogger(__name__)", HELPER)
    # (a/c) rewrite the _append_prefill_hidden_states helper: slice by this
    # forward's extend length (not len(origin_input_ids)) and route through
    # _pp_store_hidden (disk spool support). 0.5.14 factored the finished-prefill
    # append into this method, so the fix lives here now.
    src = patch(
        src,
        """    def _append_prefill_hidden_states(
        self,
        *,
        req: Req,
        logits_output: LogitsProcessorOutput,
        hidden_state_offset: int,
    ) -> int:
        req.hidden_states.append(
            logits_output.hidden_states[
                hidden_state_offset : (
                    hidden_state_offset := hidden_state_offset
                    + len(req.origin_input_ids)
                )
            ]
            .cpu()
            .clone()
            .tolist()
        )
        return hidden_state_offset""",
        """    def _append_prefill_hidden_states(
        self,
        *,
        req: Req,
        logits_output: LogitsProcessorOutput,
        hidden_state_offset: int,
        extend_input_len: int,
    ) -> int:
        # PATCH(proof-pilot): the batch hidden tensor holds only THIS forward's
        # extend tokens; slicing by the full original prompt length
        # (len(req.origin_input_ids)) misassigns hidden states whenever requests
        # are packed into one forward or chunked (upstream #8066 family). Slice by
        # this forward's extend length and route through _pp_store_hidden, which
        # writes bf16 to disk when SGLANG_HIDDEN_SPOOL_DIR is set (the default
        # list+zmq return path is ~12x slower than the forward).
        _lo = hidden_state_offset
        hidden_state_offset += extend_input_len
        _pp_store_hidden(
            req,
            logits_output.hidden_states[_lo:hidden_state_offset],
        )
        return hidden_state_offset""",
    )
    # (a) finished-prefill call site: pass this forward's extend length.
    src = patch(
        src,
        """                        hidden_state_offset = self._append_prefill_hidden_states(
                            req=req,
                            logits_output=logits_output,
                            hidden_state_offset=hidden_state_offset,
                        )""",
        """                        # PATCH(proof-pilot): pass this forward's extend length so
                        # the hidden slice matches the tokens actually in this batch.
                        hidden_state_offset = self._append_prefill_hidden_states(
                            req=req,
                            logits_output=logits_output,
                            hidden_state_offset=hidden_state_offset,
                            extend_input_len=(
                                extend_input_len_per_req[i]
                                if extend_input_len_per_req is not None
                                else req.extend_input_len
                            ),
                        )""",
    )
    # (b) being-chunked branch: accumulate intermediate-chunk hidden instead of
    # dropping. Anchor on the (unique) generation-branch chunked else block —
    # 0.5.14 renamed req.is_chunked -> req.inflight_middle_chunks.
    src = patch(
        src,
        """                else:
                    # being chunked reqs' prefill is not finished
                    req.inflight_middle_chunks -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req""",
        """                else:
                    # being chunked reqs' prefill is not finished
                    req.inflight_middle_chunks -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # PATCH(proof-pilot): intermediate prefill chunks carry hidden
                    # states too (capture_hidden_mode is FULL for the whole batch);
                    # append them so the pieces accumulate across chunks in
                    # req.hidden_states instead of only the last chunk being kept.
                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        hidden_state_offset = self._append_prefill_hidden_states(
                            req=req,
                            logits_output=logits_output,
                            hidden_state_offset=hidden_state_offset,
                            extend_input_len=(
                                extend_input_len_per_req[i]
                                if extend_input_len_per_req is not None
                                else req.extend_input_len
                            ),
                        )""",
    )
    return src


SCORE_ROUTE = '''
# PATCH(proof-pilot): OPD teacher /score — native concurrent generate (sglang scheduler does
# continuous batching) + had+int6 encode OFF the scheduler thread (here in the server process,
# via threadpool). Scheduler spools bf16 (SGLANG_HIDDEN_SPOOL_DIR); this reads it back locally and
# returns 3328 B/tok packed bytes. Keeps prefill throughput at ceiling while giving server-side quant.
_PP_SCORE_ROT = None
_PP_SCORE_HEAD = None


def _pp_load_teacher_head(model_path, hid, device):
    import json as _json
    import os as _os

    from safetensors import safe_open as _safe_open

    idx = _json.load(open(_os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]
    fn = idx["head.weight"]
    with _safe_open(_os.path.join(model_path, fn), framework="pt", device="cpu") as f:
        w = f.get_tensor("head.weight")
    if w.shape[1] != hid:
        raise RuntimeError(f"teacher head dim mismatch: got={tuple(w.shape)} hid={hid}")
    return w.to(device=device, dtype=w.dtype)


def _pp_score_encode(ret, n_ids, start, return_top1=False, out_path=None):
    import glob as _glob  # noqa: F401
    import os as _os
    import sys as _sys

    import torch as _torch
    global _PP_SCORE_ROT, _PP_SCORE_HEAD
    cdir = _os.environ.get("SGLANG_HIDDEN_CODEC_DIR")
    if cdir and cdir not in _sys.path:
        _sys.path.insert(0, cdir)
    from hidden_codec import Rotator as _Rot, encode as _enc
    hs = ret["meta_info"]["hidden_states"]
    parts = []
    for p in hs:
        t = _torch.load(p, weights_only=True) if isinstance(p, str) else _torch.as_tensor(p)
        parts.append(t if t.ndim == 2 else t.unsqueeze(0))
        if isinstance(p, str):
            try:
                _os.remove(p)
            except OSError:
                pass
    h = _torch.cat(parts, dim=0)[:n_ids][start:].to("cuda").bfloat16()
    if _PP_SCORE_ROT is None:
        _PP_SCORE_ROT = _Rot(h.shape[1], device="cuda")
    packed, scales = _enc(h, _PP_SCORE_ROT)
    pb = packed.to(_torch.uint8).cpu().numpy().tobytes()
    sb = scales.to(_torch.float16).cpu().numpy().tobytes()
    tb = b""
    if return_top1:
        model_path = _os.environ.get("OPD_TEACHER_MODEL_PATH", "/models/DeepSeek-V4-Flash")
        if _PP_SCORE_HEAD is None:
            _PP_SCORE_HEAD = _pp_load_teacher_head(model_path, h.shape[1], "cuda").bfloat16()
        chunk = int(_os.environ.get("OPD_SCORE_TOP1_CHUNK", "1024"))
        outs = []
        wt = _PP_SCORE_HEAD.T
        for i in range(0, h.shape[0], chunk):
            outs.append((h[i:i + chunk] @ wt).argmax(-1).to(_torch.int32).cpu())
        tb = _torch.cat(outs).numpy().tobytes()
    if out_path:
        # PATCH(proof-pilot,opd_v2): server-side write to shared FS (P7 fix); single source of
        # truth for the file layout is opd_v2.hidden_store.write_hidden (atomic tmp+rename).
        import sys as _sys2
        _v2 = _os.environ.get("OPD_V2_SRC")
        if _v2 and _v2 not in _sys2.path:
            _sys2.path.insert(0, _v2)
        from opd_v2.hidden_store import write_hidden as _wh
        _wh(out_path, pb, sb, int(h.shape[0]), top1=tb, hid=h.shape[1])
    return pb, sb, int(h.shape[0]), tb


@app.api_route("/score", methods=["POST"])
async def pp_score_request(raw_request: Request):
    from fastapi import Response as _Resp
    from fastapi.responses import JSONResponse as _Json
    from starlette.concurrency import run_in_threadpool
    body = await raw_request.json()
    ids = body["input_ids"]
    start = int(body.get("start", 0))
    return_top1 = bool(body.get("return_top1", False))
    out_path = body.get("out_path")
    obj = GenerateReqInput(
        input_ids=ids,
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        return_hidden_states=True,
    )
    ret = await _global_state.tokenizer_manager.generate_request(obj, raw_request).__anext__()
    pb, sb, seqlen, tb = await run_in_threadpool(
        _pp_score_encode, ret, len(ids), start, return_top1, out_path)
    if out_path:
        # opd_v2: bytes already on shared FS -> return handle metadata only (never via orchestrator).
        return _Json({"seq_len": seqlen, "packed_bytes": len(pb),
                      "scales_bytes": len(sb), "top1_bytes": len(tb)})
    headers = {"X-Seq-Len": str(seqlen), "X-Packed-Bytes": str(len(pb))}
    if tb:
        headers["X-Top1-Bytes"] = str(len(tb))
    return _Resp(content=pb + sb + tb, media_type="application/octet-stream", headers=headers)


'''


def patch_http_server(src: str) -> str:
    """Add a /score route: native generate (scheduler continuous batching) + off-thread had+int6
    encode. The route is inserted before /encode. No effect on the extract pipeline (which does not
    call /score). 0.5.14's native /v1/score route is a different path and does not collide."""
    return patch(src, '@app.api_route("/encode", methods=["POST", "PUT"])',
                 SCORE_ROUTE + '@app.api_route("/encode", methods=["POST", "PUT"])')


def patch_output_streamer(src: str) -> str:
    # (5) 0.5.14 moved the hidden emit into scheduler_components/output_streamer.py, which truncates
    # req.hidden_states to req.finished_len (the OUTPUT-token count = 1 for /score max_new_tokens=1).
    # Our extract appends ONE entry per PREFILL chunk (batch_result_processor), so that truncation drops
    # every chunk after the first for prompts > --chunked-prefill-size, silently losing the tail hidden.
    # Skip the truncation in spool mode (OPD extract); keep stock behavior for the normal decode path.
    if "\nimport os\n" not in src:
        src = patch(src, "import torch\n", "import os\nimport torch\n")
    return patch(
        src,
        """                hs = req.hidden_states
                if req.finished_len is not None:
                    hs = hs[: req.finished_len]
                self.output_hidden_states.append(hs)""",
        """                hs = req.hidden_states
                # PATCH(proof-pilot): the OPD hidden-extract spool appends one entry per PREFILL chunk;
                # finished_len (output-token count, =1 for /score) would drop all chunks after the first
                # for prompts longer than --chunked-prefill-size. In spool mode keep the full per-chunk
                # prefill hidden; only truncate the stock (non-spool) decode-hidden path.
                if req.finished_len is not None and not os.environ.get("SGLANG_HIDDEN_SPOOL_DIR"):
                    hs = hs[: req.finished_len]
                self.output_hidden_states.append(hs)""",
    )


def main():
    src_dir, out_dir = sys.argv[1], sys.argv[2]
    jobs = {
        "deepseek_v4.py": patch_dsv4,
        "scheduler.py": patch_scheduler,
        "batch_result_processor.py": patch_output_processor,
        "output_streamer.py": patch_output_streamer,
        "http_server.py": patch_http_server,
    }
    for name, fn in jobs.items():
        src = open(f"{src_dir}/{name}").read()
        assert "PATCH(proof-pilot)" not in src, f"{name} already patched"
        out = fn(src)
        with open(f"{out_dir}/{name}", "w") as f:
            f.write(out)
        print(f"patched {name}")


if __name__ == "__main__":
    main()

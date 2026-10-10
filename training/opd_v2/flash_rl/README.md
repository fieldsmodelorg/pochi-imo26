# flash_rl — FP8 quantization + `update_weights_from_disk` for the OPD rollout

Lets the OPD student rollout server **deploy with FP8 quantization** (saves VRAM, faster) while
**repeatedly loading fresh bf16 student checkpoints from disk via `update_weights_from_disk`** (sglang
re-quantizes to FP8 automatically).

- Target scope: **sglang 0.5.12.post1, 7B `stage1-v2-7b` (olmo3_sink)**; originally validated at TP=1 bf16-KV,
  and the **TP=4 + fp8-KV + SWA-ratio long-context configuration is separately validated too** (see §5.3).
- Conclusion: this path is broken in stock sglang; this directory provides a **3-edit `loader.py` bind-mount
  patch**, measured stable over 10/10 repeated reloads, bit-exact, no leak.

---

## 1. The problem: why stock doesn't work

OPD rollout is currently **bf16** (`training/opd/examples/run_rollout_service.sh`, no `--quantization`), and
`update_weights_from_disk` works. As soon as you add FP8:

| configuration | result |
|---|---|
| `--quantization fp8` (online quantization of bf16) | server OK (weights 15G->8.1G), but **the 1st `update_weights_from_disk` crashes** |
| pre-quantized compressed-tensors fp8 + update from fp8 disk | **same crash** |

**Mechanism**: at startup with `--quantization fp8`, `process_weights_after_loading` quantizes each layer's
weight **to fp8 and transposes it** (`[out,in]`->`[in,out]`) and drops the quantizing `weight_loader`. When
`update_weights_from_disk` then calls `model.load_weights()`, those params fall back to
`default_weight_loader`, which forces a bf16 `[out,in]` into the transposed fp8 `[in,out]` -> shape assert
fails -> scheduler SIGQUIT.

## 2. The fix is flash_rl, but its from_disk path is broken

sglang's built-in **`--load-format flash_rl` (`QuantizedRLModelLoader`)** is designed exactly for "serve fp8
+ reload re-quantize" (used by verl/slime RLHF). But it was only validated on
**`update_weights_from_tensor`** (co-located, GPU tensor); the **`update_weights_from_disk`** path hits a
chain of bugs on our custom Olmo3Sink in 0.5.12.post1 / 0.5.13. Traced and fixed one by one:

| # | bug | symptom | fix |
|---|---|---|---|
| 1 | `load_weights_and_postprocess` **re-wraps a proxy layer** on reload, each capturing the previous layer's proxy -> the Nth reload recurses N layers, each `list(weights)`+re-quantizes the whole model | slows down over time, **OOM after a few rounds** (looks like a leak) | on reload, call the existing proxy directly and `return`, don't re-wrap (also skips the tail postprocess that would double-quant) |
| 2 | `SKIP_QUANTIZATION_PARAMS` is Qwen2-shaped and **misses Olmo2's `q_norm`/`k_norm`/`post_feedforward_layernorm`**, dequantizing 1-D norms | `RuntimeError: Tensor match failed` (the 2-D quant kernel gets a 1-D input) | add `weight.dim() >= 2` to the quant branch; 1-D always loads as-is (keep) |
| 3 | from_disk yields a **CPU** tensor, handed directly to `per_token_group_quant_8bit` which only supports CUDA | `NotImplementedError: ... 'CPU' backend` | `weight.to(cuda)` before quantizing |

> Note: the recursion (bug 1) is the real cause of "only breaks after several rounds" — not an actual leak.
> The GPU-avail at the start of each reload is stable; after fixing the recursion the transient is fixed and
> does not grow with the round count.

The full diff of the 3 edits is in `patches/flash_rl.patch.diff` (46 lines), logically touching only two
methods of `QuantizedRLModelLoader`.

## 3. Measured validation (sglang 0.5.12.post1, H200, TP=1, stage1-v2-7b)

After applying the patch (`tests/` scripts):

- **10/10 repeated reloads succeed**, ~3.2s each; **GPU avail dead-stable at 19.8GB throughout (0 leak)**,
  even at `mem-fraction-static 0.85` (only ~20GB headroom).
- **`update_weights_from_disk` to opd == fresh-loading opd, fully bit-exact** (`tests/hashgen.py`, 4 prompt
  output hashes identical). This is the **strongest fidelity proof**: reload and a fresh load produce a
  "bit-identical" model -> **all parameters** (including `embed_tokens`/`lm_head`/each `layernorm`/`q_norm`/
  `k_norm`/`sinks`) are updated correctly, **nothing stale**.
- **deploy -> opd -> deploy round-trip bit-exact** (output hashes identical) -> re-quantization is
  deterministic and consistent with the startup path.
- **the opd checkpoint weights actually take effect** (outputs change), and generation is healthy after the
  update: multi-step arithmetic, proofs, 512-token long decodes, chat-completions (`80 km/h`, `12`) all correct.
- 0 hard errors (CPU-backend / Tensor-match / contiguity / OOM all gone).
- Passed an independent agent code review: **no must-fix** for this scope; the reviewer's concern "the
  skip-list norms don't update" was **disproven** by the fresh-vs-reload test above (see §5.1).

## 4. Usage

### Launch the FP8 rollout server
```bash
CUDA_VISIBLE_DEVICES=4 PORT=8200 ./run_rollout_fp8.sh --port 8200
# = run_rollout_service.sh + --quantization fp8 --load-format flash_rl
#   + bind-mount patches/loader.py. Weights ~8.1G (bf16 ~15G).

# long-context configuration (TP4 + fp8 KV + SWA mem, see §5.3):
CUDA_VISIBLE_DEVICES=4,5,6,7 KV_CACHE_DTYPE=fp8_e4m3 SWA_RATIO=0.5 CONTEXT_LEN=65536 \
  ./run_rollout_fp8.sh --port 8200 --tp 4
```
After that, call `/update_weights_from_disk` as usual (OPD's `RolloutClient.update_weights_from_disk(path)`
needs no change), and sglang re-quantizes the new bf16 checkpoint to fp8 automatically.

### Regenerate the patched loader when switching sglang versions
`patches/loader.py` is extracted and modified from the 0.5.12.post1 image (version-dependent). When switching
versions, re-apply by anchor strings with `apply_patch.py` (idempotent; errors clearly if an anchor doesn't match):
```bash
apptainer exec <img.sif> cat /sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py > stock.py
python apply_patch.py stock.py patches/loader.py
# or regenerate automatically at launch: REGEN_LOADER=1 ./run_rollout_fp8.sh
```

### Re-run validation
```bash
python tests/cycle.py    <port> <ckptA> <ckptB> 10      # 10 reload stress rounds + bit-exact round-trip
python tests/validate.py <port>                          # multi-prompt / long decode / round-trip / degeneration
python tests/probe.py    generate <port>                 # single-shot generate / update
# fidelity comparison: two servers (one reloaded to X, one fresh-loaded X) should output all-MATCH hashes
python tests/hashgen.py  <portA> update <ckptX>          # server A: reload to X then generate
python tests/hashgen.py  <portB>                         # server B: fresh-load X then generate
```

## 5. Scope and caveats

### 5.1 weight fidelity: all parameters update (measured, not just the quantized linears)
The `SKIP_QUANTIZATION_PARAMS` in the `_get_updated_params` / copy-back loop **only controls the "copy-back
fixup"** (restoring the original storage of fp8 linears that were transposed after quantization), and **does
not affect the actual weight loading** — the actual load is done by `first_time_load_weights(...)` inside
`rebinding_and_load_weights` (= the real `model.load_weights`, loader.py:1203), which writes the entire bf16
weight stream (including embed/lm_head/each norm) into the corresponding params. So norm/embed/lm_head **do
not go stale**. Confirmed by a fresh-load-opd vs reload-to-opd **bit-exact** test (`tests/hashgen.py`, 4/4 MATCH).

### 5.2 Other
- **TP=1 and TP=4 validated (7B); 32B untested**. Edit #3's per-rank device-move is correct; **TP>1 scale
  sharding (`_apply_scale_update`) was originally an unvalidated point, and the 2026-06-17 long-context test
  passed at TP4 (§5.3)**. 32B's reload transient is larger; leave more mem headroom and test separately.
- **`load_format` must be flash_rl**: the early-return only fires when a reload resolves to
  `load_format=flash_rl` (client passes it explicitly, or omits it and the server defaults to
  `--load-format flash_rl`). When the OPD client/orchestrator calls `update_weights_from_disk`, **do not pass
  a different `load_format`** (like `auto`), otherwise it goes through `DefaultModelLoader`, bypasses the
  whole override, and blows up. `RolloutClient.update_weights_from_disk` currently sends no `load_format`, which is correct.
- **Only applies to dense per-channel fp8**: this patch is correct for the cutlass per-channel fp8 path on
  H200/Blackwell; if you switch to block-fp8 (`weight_block_size`) or the per-tensor path on older cards, the
  scale granularity/layout won't match the startup quantization and it needs re-validation.
- **Version-dependent**: `patches/loader.py` corresponds to 0.5.12.post1; 0.5.13 has the same code in that
  section and the logic applies, but pin the image or regenerate with `apply_patch.py`.
- **Memory**: reload needs transient scratch; fp8 saves ~6G of weights vs bf16, but the reload peak consumes
  some KV headroom. 7B/TP1 at `mem-fraction 0.85` is OK.
- **Precision**: the train/deploy mismatch risk of fp8 rollout for distillation is not assessed here (this
  directory only addresses "does it run"). If you care, keeping the rollout at bf16 is still the safest
  option; fp8's VRAM/speed benefit is most worthwhile for **frozen deployment** (`deploy/quant`).
- This is a bind-mount monkeypatch, **it does not modify the image**; same style as
  `deploy/target/olmo2_sink.py` and `deploy/quant/patches/compressed_tensors.py`.
- Code-reviewed for these 3 edits (review conclusion outside the git log): correct for this scope, no must-fix.

### 5.3 long-context configuration test (2026-06-17, H200×4, TP4)
For long CoT, the rollout can enable three long-context flags simultaneously; measured **all can be enabled
and weight update works**:

| flag | env | result |
|---|---|---|
| fp8 KV cache | `KV_CACHE_DTYPE=fp8_e4m3` | ✓ KV pool reaches ~1.88M tokens. **Use e4m3**: FA3 supports it (keeps the fa3 backend), 3-bit mantissa is less lossy; `fp8_e5m2` forces the attn backend down to **triton** (slower). |
| TP4 | `--tp 4` | ✓ 32 heads÷4=8/rank; TP8 (÷8=4/rank) divides evenly and also works. For a 7B model, TP is not to fit the model but to **split the KV cache N ways for long-context capacity** (trading decode TP communication for KV headroom). |
| sliding-window mem | `SWA_RATIO=0.5` -> `--swa-full-tokens-ratio` | ✓ hybrid SWA pool enabled (olmo3 = 24 SWA : 8 full, window 4096); when the window is much smaller than ctx it can be lowered further to save more. |

**weight update (production path: `pause(in_place)` + `update_weights_from_disk(flush_cache=False)` + `continue`)**:
- 6/6 reloads `success=True`, ~3s each, no slowdown, no leak; **reload->reload bit-exact** (deploy/opd each
  the same hash every round); the engine is deterministic; the server is still alive after 6 rounds + 11.7k-tok long-context decode.
- **Closes §5.2's originally-flagged TP>1 unvalidated point**: TP4 scale sharding works.
- Note: **initial-boot quantization ≠ reload-path quantization** (under TP4, the source of validate.py's
  bit-exact "FAIL"), but training only takes the reload path, and reload->reload is bit-exact, so it's **harmless**.
- Tradeoff: fp8 KV is lossy attention -> the rollout sampling distribution differs slightly from the student's
  true distribution (a small on-policy leak), and e4m3 is more accurate than e5m2.

### 5.4 KV pool test + concurrency finalization (2026-06-20, local TP4, model=opd-v2-lc128k-softdistill-v2test-deploy)
config: `KV_CACHE_DTYPE=fp8_e4m3 SWA_RATIO=0.2 CONTEXT_LEN=131072 MEMFRAC=0.85 --tp 4`. The startup log
confirms both SWA and fp8-KV took effect (`Hybrid swa model: Olmo3SinkForCausalLM` + `Using KV cache dtype:
torch.float8_e4m3fn` + `SWAKVPool`):

| pool | tokens | KV size | layers |
|---|---|---|---|
| **full** | **4,711,012** | K+V 35.94GB each | 8 full-attention layers |
| swa | 942,202 | K+V 21.57GB each | 24 sliding-window layers (window 4096) |
| total | — | **115.0GB** (20.6GB avail left) | — |

- **Long sequences bind the full pool** (the full-attn layers store all positions of every concurrent
  sequence): `conc × avg_len ≤ 4.71M`. The swa pool takes ≤window 4096 per sequence, so 942k/4096≈230
  sequences before binding -> not the bottleneck under long CoT.
- concurrency table (per replica): avg 32k->144, 64k->72, 90k->52, 128k (fully hit)->36.
- **Finalized `ROLLOUT_MAXRUN=64`/replica**: 64×64k=4.1M<4.71M is safe; the OPD rollout average is far below
  128k -> plenty of margin. Only exceeded if all hit 128k (-> occasional preempt; under `--disable-radix-cache`
  a preempt = recompute, watch `rollout/length_rate`).
- ⚠️ **conc>10 must be paired with `--cuda-graph-max-bs`**: in the test the cuda graph only captured up to
  bs=10 (=the default MAXRUN), and bs>10 fell back to eager. `run_rollout_fp8.sh` already defaults
  `CUDA_GRAPH_MAX_BS`=`MAXRUN` (setting MAXRUN=64 auto-captures up to 64).
- swa_ratio: 0.2 is well-tuned (swa 942k supports ~230 concurrent windows, far above what's needed; full pool
  maximized). If concurrency is far below 230 you could try r=0.15 for ~9% more full pool, marginal.
- **Production run**: `examples/run_agentic_mn_32b.sbatch` bakes in this set for the 32B run (ctx130816 /
  e4m3 / swa0.2 / MAXRUN64 / teacher memfrac0.6).

## 6. Files

```
flash_rl/
├── README.md                     # this document
├── run_rollout_fp8.sh            # launch the fp8 + flash_rl rollout server (with bind-mount)
├── apply_patch.py                # regenerate the 3-edit patch from any image's loader.py (idempotent)
├── patches/
│   ├── loader.py                 # patched QuantizedRLModelLoader (bind-mount target, 0.5.12.post1)
│   └── flash_rl.patch.diff       # unified diff of the 3 edits (human-readable)
└── tests/
    ├── probe.py                  # single-shot generate / update_weights_from_disk
    ├── cycle.py                  # N reload stress rounds + bit-exact round-trip check
    ├── validate.py               # multi-prompt + long decode + degeneration + round-trip validation
    └── hashgen.py                # per-prompt output hash (for fresh-load vs reload fidelity comparison)
```

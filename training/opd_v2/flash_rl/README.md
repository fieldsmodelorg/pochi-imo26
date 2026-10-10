# flash_rl — FP8 quantization + `update_weights_from_disk` for the OPD rollout

Lets the OPD student rollout server **deploy with FP8 quantization** (saves VRAM, faster) while
**repeatedly loading fresh bf16 student checkpoints from disk via `update_weights_from_disk`** (sglang
re-quantizes to FP8 automatically).

- Target: **sglang 0.5.12.post1, `olmo3_sink` student** (validated on the 7B `stage1-v2-7b` at TP=1 bf16-KV
  and at TP=4 + fp8-KV + SWA-ratio long context, see §5.3).
- This path is broken in stock sglang; this directory provides a **3-edit `loader.py` bind-mount patch**
  (stable over repeated reloads, bit-exact, no leak) plus a small `model_config.py` patch that enables the
  hybrid SWA KV pool for Olmo3Sink.

---

## 1. The problem: why stock doesn't work

`update_weights_from_disk` works on a bf16 rollout server (no `--quantization`). As soon as you add FP8:

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
+ reload re-quantize" (used by verl/slime RLHF). But it was only exercised via
**`update_weights_from_tensor`** (co-located, GPU tensor); the **`update_weights_from_disk`** path hits a
chain of bugs on the custom Olmo3Sink in 0.5.12.post1 / 0.5.13:

| # | bug | symptom | fix |
|---|---|---|---|
| 1 | `load_weights_and_postprocess` **re-wraps a proxy layer** on reload, each capturing the previous layer's proxy -> the Nth reload recurses N layers, each `list(weights)`+re-quantizes the whole model | slows down over time, **OOM after a few rounds** (looks like a leak) | on reload, call the existing proxy directly and `return`, don't re-wrap (also skips the tail postprocess that would double-quant) |
| 2 | `SKIP_QUANTIZATION_PARAMS` is Qwen2-shaped and **misses Olmo2's `q_norm`/`k_norm`/`post_feedforward_layernorm`**, dequantizing 1-D norms | `RuntimeError: Tensor match failed` (the 2-D quant kernel gets a 1-D input) | add `weight.dim() >= 2` to the quant branch; 1-D always loads as-is (keep) |
| 3 | from_disk yields a **CPU** tensor, handed directly to `per_token_group_quant_8bit` which only supports CUDA | `NotImplementedError: ... 'CPU' backend` | `weight.to(cuda)` before quantizing |

> The recursion (bug 1) is the cause of "only breaks after several rounds" — not an actual leak. The GPU
> memory available at the start of each reload is stable; with the recursion fixed the reload transient
> does not grow with the round count.

The full diff of the 3 edits is in `patches/flash_rl.patch.diff`, logically touching only two methods of
`QuantizedRLModelLoader`.

## 3. Expected behavior (sglang 0.5.12.post1, H200, TP=1, stage1-v2-7b)

With the patch applied, the `tests/` scripts check that:

- Repeated reloads succeed (~3.2s each) with constant GPU memory available across rounds (no leak), even at
  `mem-fraction-static 0.85`.
- **`update_weights_from_disk` to checkpoint X == fresh-loading X, bit-exact** (`tests/hashgen.py`, per-prompt
  output hashes identical). This shows **all parameters** (including `embed_tokens`/`lm_head`/each
  `layernorm`/`q_norm`/`k_norm`/`sinks`) are updated, nothing stale.
- **A -> B -> A round-trip is bit-exact** -> re-quantization is deterministic and consistent with the startup path.
- The new checkpoint's weights take effect (outputs change) and generation stays healthy after the update
  (multi-step arithmetic, proofs, 512-token decodes, chat-completions).

## 4. Usage

### Launch the FP8 rollout server
```bash
CUDA_VISIBLE_DEVICES=4 PORT=8200 ./run_rollout_fp8.sh --port 8200
# = a plain sglang rollout launch + --quantization fp8 --load-format flash_rl
#   + bind-mount of patches/loader.py and patches/model_config.py. Weights ~8.1G (bf16 ~15G).

# long-context configuration (TP4 + fp8 KV + SWA mem, see §5.3):
CUDA_VISIBLE_DEVICES=4,5,6,7 KV_CACHE_DTYPE=fp8_e4m3 SWA_RATIO=0.5 CONTEXT_LEN=65536 \
  ./run_rollout_fp8.sh --port 8200 --tp 4
```
After that, call `/update_weights_from_disk` as usual (OPD's `RolloutClient.update_weights_from_disk(path)`
needs no change), and sglang re-quantizes the new bf16 checkpoint to fp8 automatically.

### Regenerate the patched files when switching sglang versions
`patches/loader.py` and `patches/model_config.py` are extracted and modified from the 0.5.12.post1 image
(version-dependent). When switching versions, re-apply by anchor strings with `apply_patch.py` /
`patches/apply_swa_patch.py` (idempotent; errors clearly if an anchor doesn't match):
```bash
apptainer exec <img.sif> cat /sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py > stock.py
python apply_patch.py stock.py patches/loader.py
apptainer exec <img.sif> cat /sgl-workspace/sglang/python/sglang/srt/configs/model_config.py > stock_mcfg.py
python patches/apply_swa_patch.py stock_mcfg.py patches/model_config.py
# or regenerate both automatically at launch: REGEN_LOADER=1 ./run_rollout_fp8.sh
```

### Run the validation scripts
```bash
python tests/cycle.py    <port> <ckptA> <ckptB> 10      # 10 reload stress rounds + bit-exact round-trip
python tests/validate.py <port>                          # multi-prompt / long decode / round-trip / degeneration
python tests/probe.py    generate <port>                 # single-shot generate / update
# fidelity comparison: two servers (one reloaded to X, one fresh-loaded X) should output all-MATCH hashes
python tests/hashgen.py  <portA> update <ckptX>          # server A: reload to X then generate
python tests/hashgen.py  <portB>                         # server B: fresh-load X then generate
```

## 5. Scope and caveats

### 5.1 Weight fidelity: all parameters update (not just the quantized linears)
The `SKIP_QUANTIZATION_PARAMS` in the `_get_updated_params` / copy-back loop **only controls the "copy-back
fixup"** (restoring the original storage of fp8 linears that were transposed after quantization), and **does
not affect the actual weight loading** — the actual load is done by `first_time_load_weights(...)` inside
`rebinding_and_load_weights` (= the real `model.load_weights`), which writes the entire bf16 weight stream
(including embed/lm_head/each norm) into the corresponding params. So norm/embed/lm_head **do not go stale**;
the fresh-load vs reload bit-exact check (`tests/hashgen.py`) confirms this.

### 5.2 Other
- **TP=1 and TP=4 validated on 7B** (TP>1 scale sharding in `_apply_scale_update` included, §5.3). Edit #3's
  per-rank device-move is correct for TP>1. 32B's reload transient is larger; leave more memory headroom.
- **`load_format` must be flash_rl**: the early-return only fires when a reload resolves to
  `load_format=flash_rl` (client passes it explicitly, or omits it and the server defaults to
  `--load-format flash_rl`). When the OPD client/orchestrator calls `update_weights_from_disk`, **do not pass
  a different `load_format`** (like `auto`), otherwise it goes through `DefaultModelLoader`, bypasses the
  whole override, and blows up. `RolloutClient.update_weights_from_disk` sends no `load_format`, which is correct.
- **Only applies to dense per-channel fp8**: this patch is correct for the cutlass per-channel fp8 path on
  H200/Blackwell; if you switch to block-fp8 (`weight_block_size`) or the per-tensor path on older cards, the
  scale granularity/layout won't match the startup quantization and it needs re-validation.
- **Version-dependent**: `patches/loader.py` corresponds to 0.5.12.post1; 0.5.13 has the same code in that
  section and the logic applies, but pin the image or regenerate with `apply_patch.py`.
- **Memory**: reload needs transient scratch; fp8 saves ~6G of weights vs bf16, but the reload peak consumes
  some KV headroom. 7B/TP1 at `mem-fraction 0.85` is OK.
- **Precision**: the train/deploy mismatch risk of fp8 rollout for distillation is not assessed here (this
  directory only addresses "does it run"). Keeping the rollout at bf16 is the safest option.
- This is a bind-mount monkeypatch, **it does not modify the image**; same style as
  `training/deploy/target/olmo2_sink.py`.

### 5.3 Long-context configuration (H200×4, TP4)
For long CoT, the rollout can enable three long-context flags simultaneously, and weight update works with
all of them:

| flag | env | notes |
|---|---|---|
| fp8 KV cache | `KV_CACHE_DTYPE=fp8_e4m3` | KV pool reaches ~1.88M tokens (TP4, 7B). **Use e4m3**: FA3 supports it (keeps the fa3 backend), 3-bit mantissa is less lossy; `fp8_e5m2` forces the attn backend down to **triton** (slower). |
| TP4 | `--tp 4` | 32 heads÷4=8/rank; TP8 (÷8=4/rank) divides evenly and also works. For a 7B model, TP is not to fit the model but to **split the KV cache N ways for long-context capacity** (trading decode TP communication for KV headroom). |
| sliding-window mem | `SWA_RATIO=0.5` -> `--swa-full-tokens-ratio` | hybrid SWA pool (olmo3 = 24 SWA : 8 full, window 4096; needs `patches/model_config.py`); when the window is much smaller than ctx it can be lowered further to save more. |

**Weight update (production path: `pause(in_place)` + `update_weights_from_disk(flush_cache=False)` + `continue`)**:
reloads succeed (~3s each, no slowdown, no leak) and **reload->reload is bit-exact**.
- **Initial-boot quantization ≠ reload-path quantization** under TP4 (so `validate.py`'s boot-vs-reload
  bit-exact check reports FAIL there), but training only takes the reload path, and reload->reload is
  bit-exact, so it's **harmless**.
- With TP>1 and several replicas per node, `run_rollout_fp8.sh` passes `--disable-custom-all-reduce`: two
  TP groups on one node can crash custom all-reduce (`custom_all_reduce.cuh: CUDA error: invalid argument`)
  under cuda-graph capture; NCCL all-reduce over NVLink has negligible cost here.
- Tradeoff: fp8 KV is lossy attention -> the rollout sampling distribution differs slightly from the student's
  true distribution (a small on-policy leak), and e4m3 is more accurate than e5m2.

### 5.4 KV pool sizing and concurrency (TP4, ctx 128k)
Config: `KV_CACHE_DTYPE=fp8_e4m3 SWA_RATIO=0.2 CONTEXT_LEN=131072 MEMFRAC=0.85 --tp 4`. The startup log
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
- Concurrency per replica: avg 32k->144, 64k->72, 90k->52, 128k (fully hit)->36.
- **`ROLLOUT_MAXRUN=64`/replica**: 64×64k=4.1M<4.71M is safe; the OPD rollout average is far below
  128k -> plenty of margin. Only exceeded if all hit 128k (-> occasional preempt; under `--disable-radix-cache`
  a preempt = recompute, watch `rollout/length_rate`).
- ⚠️ **conc>10 must be paired with `--cuda-graph-max-bs`**: otherwise the cuda graph only captures up to
  the default bs=10 and larger batches fall back to eager. `run_rollout_fp8.sh` defaults
  `CUDA_GRAPH_MAX_BS`=`MAXRUN` (setting MAXRUN=64 auto-captures up to 64).
- swa_ratio: 0.2 leaves the swa pool (942k, ~230 concurrent windows) far above what's needed while
  maximizing the full pool. If concurrency is far below 230, r=0.15 gives ~9% more full pool (marginal).
- **Production run**: `examples/run_agentic_mn_32b.sbatch` bakes in this set for the 32B run (ctx130816 /
  e4m3 / swa0.2 / MAXRUN64 / teacher memfrac0.6).

## 6. Files

```
flash_rl/
├── README.md                     # this document
├── run_rollout_fp8.sh            # launch the fp8 + flash_rl rollout server (with bind-mounts)
├── apply_patch.py                # regenerate the 3-edit loader patch from any image's loader.py (idempotent)
├── patches/
│   ├── loader.py                 # patched QuantizedRLModelLoader (bind-mount target, 0.5.12.post1)
│   ├── flash_rl.patch.diff       # unified diff of the 3 edits (human-readable)
│   ├── model_config.py           # patched sglang model_config.py: hybrid SWA KV pool for Olmo3Sink (bind-mount target)
│   └── apply_swa_patch.py        # regenerate model_config.py from any image's stock file (anchored)
└── tests/
    ├── probe.py                  # single-shot generate / update_weights_from_disk
    ├── cycle.py                  # N reload stress rounds + bit-exact round-trip check
    ├── validate.py               # multi-prompt + long decode + degeneration + round-trip validation
    └── hashgen.py                # per-prompt output hash (for fresh-load vs reload fidelity comparison)
```

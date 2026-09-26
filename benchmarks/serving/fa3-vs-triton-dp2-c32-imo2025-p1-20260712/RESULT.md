# FA3 versus Triton on the DP2 production-shaped workload

## Outcome

FA3 is the dominant optimization for this long-context, high-concurrency DFlash workload. It increased BF16 throughput from **624.49 tok/s** to **2,856.11 tok/s** and quantized throughput from **641.83 tok/s** to **3,329.66 tok/s**.

| Target and draft precision | Triton attention | FA3 attention | FA3 / Triton |
|---|---:|---:|---:|
| BF16 target + BF16 draft | 624.491 tok/s | **2,856.111 tok/s** | **4.574x** |
| Humming W4A8 target + INT4/W4A16 draft | 641.826 tok/s | **3,329.661 tok/s** | **5.188x** |

FA3 reduced the BF16 wall time from 419.773 seconds to 91.784 seconds and the quantized wall time from 408.435 seconds to 78.730 seconds. All four measurements generated exactly 262,144 completion tokens.

Once FA3 removed the attention bottleneck, quantization became materially useful:

| Attention backend | BF16 | Quantized | Quantized / BF16 |
|---|---:|---:|---:|
| Triton | 624.491 tok/s | 641.826 tok/s | 1.0278x |
| FA3 | 2,856.111 tok/s | 3,329.661 tok/s | **1.1658x** |

The earlier 2.78% quantization result was therefore not evidence that Humming was ineffective. Triton attention dominated enough of the end-to-end time to hide most of the lower-cost target and draft MLP execution. With Hopper-optimized FA3 attention, quantization adds **16.58%** throughput and saves another 13.05 seconds.

## Matched workload

Every server received the same production-shaped workload:

- MathArena IMO 2025 problem 1 and the exact ycchen generation prompt;
- 32 simultaneous requests, round-robin over TP1/DP2;
- 16 active requests per H200 replica;
- the same 32 production sample IDs and stable seeds;
- temperature 1.0, top-p 0.95, and no greedy override;
- 426 prompt tokens and an 8,192-token completion ceiling per request;
- DFlash block size 8 and draft window 512;
- BF16 persistent target and draft KV storage;
- prefix cache flushed immediately before each measurement; and
- no benchmark warm-up request.

Only the attention backend changed between each Triton result and its matched FA3 result. FA3 was applied to both target and DFlash draft attention.

## FA3 latency results

| Metric | BF16 FA3 | Quantized FA3 |
|---|---:|---:|
| Wall time | 91.784 s | **78.730 s** |
| Aggregate completion throughput | 2,856.111 tok/s | **3,329.661 tok/s** |
| P50 request latency | 81.563 s | **72.133 s** |
| P95 request latency | 90.070 s | **77.500 s** |
| Mean logged DFlash accept length | 3.082 | 3.099 |
| Requests completed at 8,192 tokens | 32/32 | 32/32 |
| Requests with final-answer content | 0/32 | 0/32 |

Acceptance did not cause the throughput difference. The two means differ by only 0.017 accepted tokens per verification cycle. The quantized FA3 improvement therefore comes from faster execution rather than a favorable acceptance artifact.

## Implication for the 22.5-minute generation budget

If the measured 8K rate stayed constant, which is optimistic because attention cost increases with context length, the 22.5-minute token budget would be approximately:

- 120,492 completion tokens per BF16 request; and
- 140,470 completion tokens per quantized request.

At the same optimistic constant rates, 262,144 tokens per request would require approximately 49.0 minutes with BF16 FA3 or 42.0 minutes with quantized FA3. FA3 transforms the operating point but does not by itself prove that the full 262K ceiling can finish within 22.5 minutes.

## Output behavior

All 64 FA3 requests reached the 8,192-token ceiling and emitted reasoning content, but none emitted final-answer content. FA3 fixes inference performance; it does not fix the model's failure to terminate naturally on this prompt.

## Runtime validation

The BF16 and quantized logs both report `attention_backend='fa3'` and `speculative_draft_attention_backend='fa3'`. The quantized log additionally records:

- the mandatory target-only Humming preflight;
- 256 Humming target-layer instances across DP2;
- 32 INT4/W4A16 draft-layer instances across DP2;
- fused DFlash draft-KV materialization on both replicas; and
- BF16 persistent KV allocation with no fallback or retraction.

The initial BF16 log contains a rejected client attempt to the wrong non-`/v1` route. Every such request returned HTTP 404 before inference. The cache was flushed again before the successful measured workload, so those requests are excluded from all timing and output totals.

## Artifacts

- `comparison.json`: machine-readable four-way comparison;
- `bf16-fa3-result.json` and `quantized-fa3-result.json`: summary measurements;
- `bf16-fa3-requests.json` and `quantized-fa3-requests.json`: all per-request records; and
- `bf16-fa3-server.log` and `quantized-fa3-server.log`: complete runtime logs.

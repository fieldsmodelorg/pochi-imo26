# BF16 DFlash: FA3 versus Triton

## Outcome

FA3 increased BF16 throughput from **624.49 tok/s** to **2,856.11 tok/s**, a **4.574x** speedup on this workload. FA3 was applied to both target and DFlash draft attention.

| Attention backend | Wall time | Aggregate completion throughput |
|---|---:|---:|
| Triton + DFlash | 419.773 s | 624.491 tok/s |
| FA3 + DFlash | 91.784 s | **2,856.111 tok/s** |

Both measurements generated exactly 262,144 completion tokens: 32 requests with 8,192 tokens each.

## Matched workload

Both configurations used the BF16 target and BF16 DFlash draft with the same workload:

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

| Metric | BF16 FA3 |
|---|---:|
| Wall time | 91.784 s |
| Aggregate completion throughput | 2,856.111 tok/s |
| P50 request latency | 81.563 s |
| P95 request latency | 90.070 s |
| Mean logged DFlash accept length | 3.082 |
| Requests completed at 8,192 tokens | 32/32 |
| Requests with final-answer content | 0/32 |

## Runtime validation

The BF16 FA3 log reports `attention_backend='fa3'` and `speculative_draft_attention_backend='fa3'`.

The initial BF16 log contains a rejected client attempt to the wrong non-`/v1` route. Every such request returned HTTP 404 before inference. The cache was flushed again before the successful measured workload, so those requests are excluded from all timing and output totals.

## Scope

These historical measurements, recorded on July 12, 2026, use the earlier `opd-32b-deploy` checkpoint on two H200s with TP1/DP2. There is one timed 32-request batch per condition. All requests reached the 8,192-token ceiling; the measurements do not assess proof quality or end-to-end proof-search speed on the final eight-GPU deployment.

## Artifacts

- [FA3 versus Triton comparison](fa3-vs-triton-comparison.json).
- Triton + DFlash: [summary](triton-dflash/result.json), [per-request measurements](triton-dflash/requests.json), [server log](triton-dflash/server.log).
- FA3 + DFlash: [summary](fa3-dflash/result.json), [per-request measurements](fa3-dflash/requests.json), [server log](fa3-dflash/server.log).

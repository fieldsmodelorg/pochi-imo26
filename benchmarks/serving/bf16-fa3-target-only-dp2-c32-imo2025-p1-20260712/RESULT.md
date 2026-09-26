# BF16 FA3 target-only versus DFlash

## Outcome

On the matched 32-request production-sampling workload, BF16 DFlash more than
doubled the throughput of the same BF16 target running alone with FA3.

| BF16 mode | Wall time | Aggregate completion throughput |
|---|---:|---:|
| FA3 target-only | 188.380 s | 1,391.571 tok/s |
| FA3 + DFlash | 91.784 s | **2,856.111 tok/s** |

DFlash provided a **2.0524x throughput speedup** and reduced end-to-end wall
time by **51.28%**. This result means the draft model remains worthwhile even
if there is not enough time to quantize either checkpoint.

## Matched workload

The two measurements used:

- the BF16 `/workspace/models/opd-32b-deploy` target;
- FA3 attention on H200;
- TP1/DP2 across two GPUs, with round-robin request placement;
- 32 simultaneous requests, giving 16 active requests per replica;
- MathArena IMO 2025 problem 1 and the exact ycchen generation prompt;
- the same 32 production sample IDs and stable seeds;
- temperature 1.0, top-p 0.95, and no greedy override;
- 426 prompt tokens and an 8,192-token completion ceiling per request;
- BF16 persistent KV storage, memory fraction 0.84, and deterministic inference;
- a prefix-cache flush immediately before measurement; and
- no benchmark warm-up request.

The only material inference difference was speculative decoding: the
target-only server reported `speculative_algorithm=None`, while the comparison
server used the BF16 DFlash draft with block size 8 and draft window 512.

## Result validity

All 32 target-only requests completed successfully with exactly 8,192 tokens
and finish reason `length`, for 262,144 measured completion tokens. All emitted
reasoning content and none emitted final-answer content. The server log confirms
`attention_backend='fa3'`, `tp_size=1`, `dp_size=2`, BF16 KV allocation, no
speculative algorithm, and successful cache flushing on both replicas. It
contains no CUDA out-of-memory error, runtime exception, or NaN report.

## Interpretation

FA3 accelerates attention, but it does not eliminate the cost of executing all
64 target layers once per output token. Target-only autoregressive decoding
must pay that full target pass for every token. DFlash proposes blocks with a
smaller draft model and verifies them with the target, amortizing target work
across multiple accepted tokens. On this workload, that reduction in target
passes is larger than the added draft and verification cost, yielding the
measured 2.05x gain.

The result supports a practical near-term configuration of **BF16 target +
BF16 draft + FA3 + DFlash**. Quantization is an additional optimization, not a
prerequisite for meeting the current throughput objective.

## Artifacts

- `result.json`: target-only summary;
- `requests.json`: all target-only per-request measurements and output hashes;
- `config.yaml`: exact strict evaluation configuration used to launch the server;
- `server.log`: complete target-only SGLang runtime log;
- `client.log`: client summary output; and
- `comparison.json`: machine-readable target-only versus DFlash comparison.

The matched DFlash source artifacts are in the adjacent
`fa3-vs-triton-dp2-c32-imo2025-p1-20260712` result directory.

# training/docker/launch — OPD v2 launchers for the cu128 image

Faithful ports of Yi-Chia's `training/opd_v2/examples/run_mn.sh` orchestration (same 4-process loop,
launch order, health gate). The launchers run from this checkout (`opd_v2.sbatch` and `launch_ssh.sh` use it
directly); an image built from this repo also bakes them at `/opt/opd/launch/`. The published run image
(`sha256:908516a7…`) predates them, so with that image always launch from the checkout.

| file | what |
|---|---|
| `opd_v2.sbatch` | **8-node slurm job, Docker image via pyxis/enroot.** Sources a preset, then runs `run_mn_cu128.sh` with every step wrapped in the container. |
| `run_mn_cu128.sh` | multi-node slurm launcher (her V33 topology). Runs on the batch host; `SRUN_CONTAINER_ARGS` (set by the sbatch) puts each srun step in the image. |
| `launch_ssh.sh` | **no scheduler**: starts the image on every host of a hostfile over ssh + docker, each running `run_mn_nodes.sh`. |
| `run_mn_nodes.sh` | per-node launcher: run the same script on every node with `NODE_RANK`/`NNODES`/`LAUNCH_ID`; each node runs only its role. Scheduler-free port of the launcher our B200 run used on Ai2's Beaker. |
| `env_max_b200.sh` | **our 64× B200 run**: her V33 + 160k rollout (128k → 160k). B200/B300 only. Sources `env_v33.sh`. |
| `env_v33.sh` | her V33 production config (128k) — the faithful reproduction. **H200 and B200.** |
| `run_1node.sh` | single-node 8-GPU integration launcher (co-located, no srun). |
| `env_1node_smoke.sh` | reduced 8-GPU (4:2:2) agentic smoke, 57k, 20 steps. Sized for H200. |
| `env_4gpu_olmo3.sh` | 4-GPU Olmo3→Olmo3 self-distill check of the hidden-extract + `W_rot` codec (JSD ≈ 0). |

The only differences from her scripts are the container substitutions: `apptainer exec $SIF` → the baked
`/opt/opd/opd_serve/{run_teacher,run_rollout}.sh` (sglang 0.5.14 in `/opt/venv/serve`, rollout
`--attention-backend triton`), her `$REPO/.venv` python → the image train python, and
`ATTN_IMPL=olmo3_sink_fa2`.

## Which preset on which GPU

| preset | H200 (141 GB) | B200 (180 GB) | notes |
|---|:---:|:---:|---|
| `env_v33.sh` | ✅ | ✅ | her numbers; trainer at the ~139 GB H200 edge |
| `env_max_b200.sh` | ❌ | ✅ | trainer ~174 GB/GPU; warns if sourced on a non-sm_100 node |

Nothing else is set per GPU: `run_teacher.sh` picks the MoE backend from the compute capability
(sm_100 → `flashinfer_mxfp4`, else `auto` → marlin) and the JIT cache is scoped per arch.
Her V33 ran on 8× H200 with her cu13 stack; the B200 run used this image with `env_max_b200.sh`.

## Multi-node (8 nodes × 8 GPU)

Topology: 1 teacher (TP4×2) + 4 rollout (TP4×2 fp8 = 8 replicas) + 3 trainer (FSDP2 world 24); the
orchestrator runs on the first trainer node. Node roles come from `scontrol show hostnames
$SLURM_JOB_NODELIST` (teacher = node[0], rollout = nodes[1:5], trainer = nodes[5:8]); `ROLLOUT_NNODES` is
the one balance knob (`5` → 1+5+2, her pre-V33 fallback, if `perf/rollout_starved_frac` spikes).

**With pyxis/enroot** (Docker image on plain slurm):

```bash
enroot import -o /shared/ycchen-opd+cu128.sqsh \
  'docker://chankhavu/ycchen-opd@sha256:908516a710f3f6c4157a92c0f9723ff862f84a9c3bf347601c6b2b9ca0ef1a25'   # once, pinned
OPD_IMAGE=/shared/ycchen-opd+cu128.sqsh OPD_ENV=env_max_b200.sh \
RUN_DIR=/shared/runs/opd_max_b200 JIT_CACHE_DIR=/shared/jit_cache \
STUDENT_PATH=/shared/models/student-deploy DEEPSEEK_V4_FLASH=/shared/models/DeepSeek-V4-Flash \
NCCL_IB_HCA=... NCCL_SOCKET_IFNAME=... \
sbatch training/docker/launch/opd_v2.sbatch          # OPD_ENV=env_v33.sh for H200 / the faithful 128k run
```

The sbatch bind-mounts `RUN_DIR`, `JIT_CACHE_DIR` and the model dirs at the same paths inside the
container (add more with `EXTRA_CONTAINER_MOUNTS`, extra pyxis flags with `EXTRA_CONTAINER_ARGS`).

**Without a scheduler** (ssh + docker; hostfile lines in rank order: teacher, 4 rollout, 3 trainer):

```bash
OPD_IMAGE=chankhavu/ycchen-opd@sha256:908516a710f3f6c4157a92c0f9723ff862f84a9c3bf347601c6b2b9ca0ef1a25 \
OPD_ENV=env_max_b200.sh RUN_DIR=/shared/runs/opd_max_b200 JIT_CACHE_DIR=/shared/jit_cache \
STUDENT_PATH=/shared/models/student-deploy DEEPSEEK_V4_FLASH=/shared/models/DeepSeek-V4-Flash \
bash training/docker/launch/launch_ssh.sh hosts.txt          # ... hosts.txt stop  to tear down
```

Each node publishes its hostname to `RUN_DIR` and waits for the others (the rank → host list `scontrol` gives
on slurm). Teardown uses per-launch sentinels in `RUN_DIR`: the head writes `STOP_<id>` when the orchestrator
exits, and a node whose server or trainer dies writes `FAILED_<id>_rank<r>`, which stops every other node and the
orchestrator. Sentinels are keyed by the launch, so relaunching into the same `RUN_DIR` (resume) is safe.
With your own scheduler, run `run_mn_nodes.sh` inside the image on every node with `NODE_RANK`, `NNODES`, a shared
`LAUNCH_ID` and `RUN_DIR` set (plus `NODE_HOST` if `hostname -f` isn't the name peers should use).

**Already inside an image built from this repo** (an allocation whose steps run in the container):

```bash
source /opt/opd/launch/env_v33.sh        # or env_max_b200.sh
bash   /opt/opd/launch/run_mn_cu128.sh
```

### Cluster requirements & first-run caveats

- **Slurm only:** the launcher is `srun`-based (`SLURM_JOB_ID`, `SLURM_JOB_NODELIST`, `scontrol`, per-role
  `srun --overlap --nodelist …`).
- **Shared filesystem** (same path on all 8 nodes): `RUN_DIR` is the single source of truth —
  `config.json`, `trainer_endpoint.json`, the teacher-written **hidden-state** files, the weight-sync
  **weights** buffer, DCP **checkpoints**, the agentic **pool/seed.jsonl**, and all logs. Keep `RUN_DIR`
  distinct per preset (128k and 160k pools must not mix). `JIT_CACHE_DIR` should be a FIXED shared path so
  the compile cache persists across runs.
- **Node-local, must NOT be shared:** the teacher hidden **spool** `/dev/shm/opd-v2-tea-$port` (needs a
  large `/dev/shm`) and `TRITON_CACHE_DIR=/tmp/triton_opdv2_$SLURM_NODEID`.
- **Required env:** `STUDENT_PATH`, `STUDENT_DEPLOY_PATH`, `ROLLOUT_MODEL`, `DEEPSEEK_V4_FLASH`, `RUN_DIR`
  (the sbatch defaults the middle two to `STUDENT_PATH`). `DEEPSEEK_V4_FLASH` otherwise silently falls back
  to `/models/DeepSeek-V4-Flash` and the teacher fails ~20 min into cold start.
- **Ports** (base + `PORT_SHIFT`, +i/replica): teacher HTTP `8100+` / dist `38100+` / NCCL `38600+`,
  rollout HTTP `8200+`, trainer rdzv `29500+` + HTTP `8300+`. Teacher/rollout HTTP must be reachable from
  the head node (health gate); trainer rdzv + NCCL across the 3 trainer nodes. Set `NCCL_IB_HCA` /
  `NCCL_SOCKET_IFNAME` for your fabric.
- **First-run seeding:** the orchestrator auto-seeds the pool from the public `SEED_SOURCE` at startup
  (no `HF_TOKEN` needed), so the **head node needs network egress**. To skip that, pre-seed once on a
  connected box: `python -m opd_v2.agentic.seed --run-dir $RUN_DIR`.
- **Resume:** presets default to `RESUME=0` (fresh run). After a requeue set `RESUME=1` with the same `RUN_NAME`
  / `RUN_DIR` (pool + DCP checkpoints persist).

## Single-node integration test (8× H200, 4:2:2)

```bash
export STUDENT_PATH=/scratch/.../student-deploy   # deploy-format; works for trainer + rollout
export DEEPSEEK_V4_FLASH=/scratch/.../DeepSeek-V4-Flash
export RUN_DIR=/scratch/.../runs/opd_1node_smoke
source /opt/opd/launch/env_1node_smoke.sh
bash   /opt/opd/launch/run_1node.sh
```
Layout: teacher DeepSeek TP4 [GPU 0-3] · rollout TP2 [4-5] · trainer world 2 [6-7] · orchestrator (CPU).
Default is the **agentic** path (57k, the same producer + dsflash dataset as production, `MAX_STEPS=20`).
For a lighter plumbing check without network, use `PRODUCER=single_round` (in-repo
`training/distill_gen/problems/problems.parquet`) — see the comments in `env_1node_smoke.sh`.

## What "healthy" looks like (wandb / stdout)

`train/loss` decreases, `onpolicy/weight_version` climbs (weight sync works),
`perf/rollout_starved_frac` is low (trainer isn't waiting on rollouts), `rollout/length_rate` stays near
0 (little window truncation), and on `g4/*` steps `g4/top1` rises while `learn/reverse_kl` falls
(student converging to the teacher). See [`../../docs/OPD_V2_ALGORITHM.md`](../../docs/OPD_V2_ALGORITHM.md).

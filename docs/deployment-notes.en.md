[中文](deployment-notes.md) · **English**

# DeepSeek-V4-Flash-0731 on Two GX10 Boxes

This documents a deployment that is already running: **two NVIDIA GB10 (DGX Spark / GX10) nodes, TP=2, DSpark speculative decoding at k=5, NVFP4 KV cache**.

It covers how to install it, how to verify the install worked, what performance to expect, how to locate problems, and which tuning attempts were tried and what the verdicts were. Every number was measured on this hardware; the measurement method is stated alongside.

> ## ⚠️ This document measures the hardware's limits, not the model's quality
>
> **There is no capability evaluation anywhere in it** — no benchmark scores, no output-quality comparisons, no verdict on whether this model is any good. Everything here answers one question: **how much this hardware plus this configuration can sustain, where it tops out, and what makes it silently run at half speed.**
>
> That explains how §5.1's four test prompts were chosen — by **acceptance-rate tier**, not by capability dimension. Their job is to span the range from "highly predictable" to "free-form text" so that throughput numbers become interpretable. **Using them to judge model capability would be meaningless.**

**Two more things to know before reading:**

- This configuration optimizes **single-stream latency** (agent workloads), not high-concurrency throughput. Comparing absolute numbers against a four-node non-speculative throughput setup is meaningless.
- The biggest risk here is not a crash but **"it runs, output quality is fine, and it's half as fast."** A substantial part of this document is about catching that class of failure.

---

## Contents

1. [What this deployment is](#1-what-this-deployment-is)
2. [Read this first: the throughput model](#2-read-this-first-the-throughput-model)
3. [Deployment steps](#3-deployment-steps)
4. [Parameter reference](#4-parameter-reference)
5. [Baselines and capacity planning](#5-baselines-and-capacity-planning)
6. [Measurement discipline](#6-measurement-discipline)
7. [Failure modes](#7-failure-modes)
8. [Tuning record: what was tried, what the verdict was](#8-tuning-record-what-was-tried-what-the-verdict-was)
9. [Client integration notes](#9-client-integration-notes)
10. [Methodology](#10-methodology)
11. [References](#11-references)

---

## 1. What this deployment is

### 1.1 Hardware and environment

| | head (rank 0) | worker (rank 1) |
|---|---|---|
| Management net | 192.168.1.32 | 192.168.1.33 |
| RoCE rail A | 192.168.100.1 | 192.168.100.2 |
| RoCE rail B | 192.168.101.1 | 192.168.101.2 |

- **NVIDIA GB10** (SM121, capability 12.1), 128 GB unified memory × 2
- Ubuntu 24.04.4 aarch64, kernel 6.17.0-1031-nvidia
- Driver 580.173.02, CUDA 13.0, ConnectX-7 firmware 28.45.4028
- **One QSFP cable, plugged into Port 1** between the nodes

### 1.2 Software stack

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731`, 155.43 GiB / 48 shards |
| Runtime | vLLM `0.21.1rc1.dev339+g1967a5627bc3` |
| Image | `vllm-dspark-runtime:dspark-nvfp4-stage-c` (**must include Patch A**) |
| Container | `dsv4f-vllm-dspark-1` (same name on both nodes) |
| Parallelism | TP=2, PP=1, two-node mp executor |
| Speculative decoding | DSpark, k=5, `draft_sample_method: probabilistic` |
| KV cache | `nvfp4_ds_mla`, block-size 256 |
| Context | 524,288 tokens |
| Endpoint | `http://<head>:8078/v1`, model name `deepseek-v4-flash` |

### 1.3 The production configuration quadruple

These four values are this deployment's core divergence from the upstream template. **Any performance number must be reported together with them:**

```
gmu 0.7935 / cudagraph accounting 1 / fused-markov 1 / DRAFT_CAPTURE_SIZES auto
```

As environment variables:

```bash
GPU_MEMORY_UTILIZATION=0.7935
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1
VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto
```

One command verifies all of it, including cross-node consistency:

```bash
./scripts/dsv4f-launch.sh --check-only
```

---

## 2. Read this first: the throughput model

**There is exactly one formula that makes every performance number in this document interpretable:**

```
single-stream throughput = step rate × tokens accepted per step
```

- **Step rate** is nearly constant and depends only on concurrency: about **13.5–14.5 steps/sec** single-stream, about **6.3–8.5 steps/sec** at c6 (bigger batches, slower steps)
- **Tokens accepted per step** ("acceptance length") is determined entirely by **content**, and is independent of concurrency

Measured across three content tiers, same machine, same boot:

| Content type | Acceptance length | Single-stream tok/s | Implied step rate |
|---|---|---|---|
| Counting (`Count from 1 to 300`) | 5.96 / 6 | 85.1 | 14.3 steps/sec |
| Code generation | 5.41 | 71.6 | — |
| Prose reasoning | 2.36–2.53 | 35.9–37.5 | 15.2 / 14.8 steps/sec |

**Single-stream throughput ranges from 36 to 85 tok/s — a 2.3× spread that comes entirely from acceptance rate, while GPU step rate does not move.**

### Three consequences

**① A throughput number without an acceptance length is uninterpretable.**
Seeing "36 tok/s" tells you nothing about whether the system slowed down or the content is hard to predict. **Always report both.**

**② Sizing a new workload needs no new benchmark run.**

```
single-stream tok/s ≈ 14.5 × acceptance length for that content
c6 aggregate        ≈ single-stream × 2.7
```

**③ Acceptance length is the best health signal you have.**
Speculative decoding is verified by the target model, so **a bad draft can only make you slower, never wrong**. Falling acceptance with normal output quality is therefore itself the diagnosis. See [7.1](#71-the-core-risk-runs-fine-quality-fine-half-the-speed).

---

## 3. Deployment steps

### 3.1 Fix the network before anything else

**Do not skip this.** GB10 has a known failure where RDMA bandwidth clamps at **13 Gb/s** against a nominal 200. When it happens the model still starts, still emits tokens, and is merely slow — **it surfaces no explicit error at all**.

Symptoms and the full elimination path are in [7.4](#74-network-clamped-at-13-gbs). Here, just the fix and the acceptance test:

**Fix**: finalize the cabling → stop touching it → **reboot both nodes** → re-measure.

**Acceptance test (threshold 184 Gb/s):**

```bash
# two terminals on the worker
ib_write_bw -d rocep1s0f1   -x 3 -F --report_gbits -D 15 -p 18515
ib_write_bw -d roceP2p1s0f1 -x 3 -F --report_gbits -D 15 -p 18516

# one command on the head, driving both domains at once
( ib_write_bw -d rocep1s0f1   -x 3 -F --report_gbits -D 15 -p 18515 192.168.100.2 | awk '/^ [0-9]/{print "railA="$4}' & \
  ib_write_bw -d roceP2p1s0f1 -x 3 -F --report_gbits -D 15 -p 18516 192.168.101.2 | awk '/^ [0-9]/{print "railB="$4}' & \
  wait )
```

Healthy result: **98.01 + 98.01 = 196.02 Gb/s**.

> **You must drive both domains simultaneously.** One physical QSFP port is fed half-and-half by two independent PCIe Gen5 x4 domains, each capped near 126 Gb/s. **A single `ib_write_bw` or `iperf3` will never show more than about 98** — that is not a fault.

**Persisting the network config** — `/etc/netplan/40-cx7.yaml` (head; change the last octet to .2 on the worker):

```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f1np1:
      dhcp4: no
      addresses: [192.168.100.1/24]
      mtu: 9000
    enP2p1s0f1np1:
      dhcp4: no
      addresses: [192.168.101.1/24]
      mtu: 9000
```

> ### Operational rule
> **Any time the QSFP cable is disturbed, reboot both nodes and re-run the dual-interface bandwidth test.**
> This bug is silent, reproducible, and costs 7.5× the bandwidth.

### 3.2 Configuration file

Copy `config/env.dspark.example` to `.env.dspark` in your deployment directory. At minimum change:

```bash
HF_TOKEN=<your HuggingFace token>
WORKER_HOST=192.168.100.2          # worker's RoCE address
MASTER_ADDR=192.168.100.1          # head's RoCE address
VLLM_HOST_IP=192.168.100.1
WORKER_VLLM_HOST_IP=192.168.100.2
WORKER_DIR=/home/<user>/services/dsv4f
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
```

Full parameter list in [section 4](#4-parameter-reference).

### 3.3 Apply Patch A

Patch A gives the DSpark drafter cudagraph capture sizes independent of the target model. **Measured +6% single-stream — the only solid performance gain of the tuning round.**

**Do not use a bind-mount.** The upstream launch script says so directly:

> DSpark source patches ship inside the runtime image (`recipe/overlay/`), **not as runtime bind-mounts**.

Overwrite the overlay source and let the build chain rebuild:

```bash
cp patches/dspark_proposer.py \
   ~/services/dsv4f/recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py
```

The launch script's overlay staleness check fires automatically and **builds on both nodes**:

```
verify-overlay-sources.sh → overlay image → stage-a → b → c
        ↓
build-dspark-vllm-runtime.sh defaults to WORKER_BUILD=1:
  rsync -az --delete the whole repo to the worker → worker rebuilds too
```

About one minute per node. **Cross-node consistency is guaranteed by design; no manual sync required.**

> ### ⚠️ Installing the patch is not enough — the switch must be set too
>
> ```bash
> VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto
> ```
>
> The patch's own parser reads: `""/0/off -> None (feature off, byte-identical legacy behaviour)`. **Without this variable, the patch behaves identically to stock, byte for byte.**
>
> The variable is also **absent** from upstream's `docker-compose.dspark.yml`, so writing it only into `.env.dspark` never reaches the container. Add to the `environment:` block:
>
> ```yaml
> VLLM_DSPARK_DRAFT_CAPTURE_SIZES: "${VLLM_DSPARK_DRAFT_CAPTURE_SIZES:-}"
> ```
>
> Empty means off — a safe default.

### 3.4 Launch

```bash
# 1. Confirm no leftover containers (never docker restart/start an old one)
docker ps -a | grep dspark

# 2. Drop page cache on both nodes (ssh needs -t)
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
ssh -t <worker> "sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"

# 3. Launch (the script starts worker first, head second)
cd "$REPO_DIR" && ./dsv4f-launch.sh      # REPO_DIR is the deployment directory, default ~/services/dsv4f
```

**Cold start takes about 8m30s** (first run includes JIT compilation; roughly 350s once caches exist).

### 3.5 Verify: six self-checks

`./scripts/dsv4f-launch.sh --check-only`. **Deployment succeeded only when all six are green** — each maps to a known silent failure:

| Check | Expected | Cost of failure |
|---|---|---|
| Patch 3 (both nodes) | `grep -c is_prefill_chunk …/sched/scheduler.py` = **5** | 11/12 cold prefills fail; hot requests all pass (smoke tests miss it) |
| Patch 4 (both nodes) | `grep -c shared_experts.gate_up_proj …/spec_decode/dspark.py` ≥ **2** | Acceptance 60.2% → 25.7%, 55.4 → 32.7 tok/s |
| B12X MoE (both nodes) | `VLLM_USE_B12X_MOE` = **1** | Silently falls back to DEEPGEMM_MXFP4, 55+ → 29 tok/s |
| Proposer md5 matches across nodes | Identical | **Divergent code makes collective communication desynchronize — NCCL hangs** |
| Patch A switch state | With `auto`, the `drafter-private cudagraph` activation log must be present | Patch never made it into the image |
| Acceptance length gauge | **≥ 3.5** | Any of the first three failures shows up here |

> **The md5 criterion applies to the file inside the image, not the image ID.** Each node runs its own `docker build`, so image IDs are inherently irreproducible; sources are forced identical by `rsync -az --delete`. Comparing image IDs produces a false "inconsistent" alarm.

The self-check also echoes the configuration quadruple and the gmu value suggested by that boot's log.

---

## 4. Parameter reference

### 4.1 Divergences from the upstream template

| Parameter | Template | This deployment | Rationale |
|---|---|---|---|
| `GPU_MEMORY_UTILIZATION` | 0.85 | **0.7935** | Upstream's campaign found 0.80 is the physical edge. **Note the frame of reference**: with cudagraph accounting on, 0.7935 is equivalent to 0.786 with accounting off — still inside the edge. 0.85 is a leftover that never caught up with the campaign |
| `MAX_MODEL_LEN` | 1048576 | **524288** | 1M context gives 16-minute TTFT, unusable for agents |
| `MAX_NUM_SEQS` | 12 | **6** | Leaves KV headroom alongside a 512K context |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 0 | **1** | CUDA-graph memory counted during KV allocation, less OOM risk at the edge. **gmu must be raised at the same time** or the KV pool shrinks instead (see [8.2](#82-gmu-and-cudagraph-accounting-must-change-together)) |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | 0 | **1** | Avoids materializing vocabulary-sized Markov logits. Measured neutral; kept for the theoretical bandwidth saving |
| `VLLM_DSPARK_DRAFT_CAPTURE_SIZES` | absent | **auto** | Patch A's switch; parses to `[1,2,4,6]`. **+6% single-stream** |

### 4.2 Four things not to touch

| Item | Reason |
|---|---|
| `--block-size 256` | The V4 FlashMLA + indexer kernels require **exactly 256**; 128 and 512 both fail |
| `--generation-config vllm` | Ignores the model's bundled `generation_config.json`, whose `repetition_penalty=1.05` is a documented DSpark crash risk (illegal memory access) |
| Leave `--attention-backend` unset | Stay on AUTO. `FLASHINFER_MLA_SPARSE_DSV4` does not exist in this image |
| Leave `VLLM_USE_V2_MODEL_RUNNER=1` unset | Incompatible with DSpark; rejected at startup |

### 4.3 k=5 is a hard constraint

Three independent lines of evidence. **"Lower k to be safe" is explicitly disproven:**

1. **Upstream campaign measurements**: k=4 gives a −6.1% battery mean (count −11.5%, tool −10%). Acceptance rate does climb to 66%, but fewer tokens are accepted per step. k=3 was deleted as "strictly worse" by the same mechanism
2. **Image-level constraint**: the DSpark branch of `SpeculativeConfig.hf_config_override` sets `n_predict = dspark_block_size = 5`. k=7 is rejected at startup by a divisibility check; bypassing it crashes on first generation with `The size of tensor a (7) must match the size of tensor b (5)`. The rule is **k ≤ 5, or a multiple of 5**
3. **Per-position acceptance on this machine**: position 5 still returns 0.737 — clearly still paying for itself

> DeepSeek's official model card recommends `num_speculative_tokens: 7`, but that is a property of the model, not a capability of this runtime.

### 4.4 Other key parameters (template defaults, unchanged)

```bash
MTP_NUM_TOKENS=5
MAX_NUM_BATCHED_TOKENS=8192

# NCCL — see the GID_INDEX note below
NCCL_IB_MERGE_NICS=1
NCCL_IB_GID_INDEX=3
NCCL_CROSS_NIC=1
NCCL_NET=IB
NCCL_CUMEM_ENABLE=0
NCCL_NVLS_ENABLE=0

VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_B12X_MOE=1                  # critical: =0 silently drops to 29 tok/s
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_LOCAL_ARGMAX=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
VLLM_DSV4_B12X_COMPRESSED_MLA=0
VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0

VLLM_ENGINE_READY_TIMEOUT_S=3600
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
TORCH_CUDA_ARCH_LIST=12.1a
FLASHINFER_CUDA_ARCH_LIST=12.1a
FLASHINFER_DISABLE_VERSION_CHECK=1
DG_JIT_USE_NVRTC=0
DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> **Two dead settings**: `VLLM_TRITON_MLA_SPARSE` and `VLLM_SKIP_INIT_MEMORY_CHECK` have **zero references** anywhere in the vLLM package. Setting them does nothing. Harmless to leave, but do not assume they control anything.

**`NCCL_IB_GID_INDEX=3` must not be mis-copied**: 3 = RoCE v2 + IPv4. Indexes 2 and 3 hold identical GID values and differ only in type, so getting it wrong silently falls back to RoCE v1. Verify:

```bash
cat /sys/class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/3   # should print "RoCE v2"
```

### 4.5 Full serve command line

```bash
dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 --port 8078 \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 524288 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.7935 \
  --enable-prefix-caching \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":false}' \
  --generation-config vllm \
  --enable-flashinfer-autotune \
  --nnodes 2 --node-rank ${NODE_RANK} \
  --master-addr 192.168.100.1 --master-port 25000 \
  ${HEADLESS:+--headless}
```

---

## 5. Baselines and capacity planning

### 5.1 The four standard tests

`scripts/bench-baseline.py` uses these four prompts, **frozen verbatim in the script** — change the prompt and it is no longer the same baseline, because acceptance is content-driven.

| Test | Prompt | Workload it represents |
|---|---|---|
| `count` | Count from 1 to 300, separated by commas. | Synthetic best case, highly predictable |
| `struct` | Output a JSON array of 40 objects, each with fields id, name, email, active. | Structured output |
| `code` | Write a Python function that implements a red-black tree with insert and delete. | Code generation, tool-call arguments |
| `prose` | Write a 400-word essay on why distributed systems are hard. | Free-form reasoning |

### 5.2 Single-stream baseline

**Production configuration, after 5×500-token warmup, `stream:false`, t=0, median of 3 runs:**

| Test | This machine | Upstream reference |
|---|---|---|
| count | **85.5** | 78.4 |
| struct | **74.6** | 66.1 |
| code | **71.6** | 62.2 |
| prose | **36.8** | 37.8 |

Reproduce:

```bash
python3 scripts/bench-baseline.py warmup   # 5 rounds × ~500 tokens, reach steady state
python3 scripts/bench-baseline.py bench    # the four tests
```

**Acceptance versus upstream:**

| | This machine | Upstream reference |
|---|---|---|
| Real-workload acceptance length | **5.19 / 6** | 4.0–5.0 |
| Real-workload acceptance rate | **83.8%** | 56.1% |
| Per-position acceptance | 0.937 / 0.874 / 0.853 / 0.789 / **0.737** | 0.826 / 0.725 / 0.572 / 0.471 / **0.399** |

Every position is clearly above upstream; position 5 at 0.737 vs 0.399 is the direct evidence that k=5 still pays.

### 5.3 Concurrency capacity

`scripts/loadtest.py`, again median of 3 rounds. **Aggregate throughput, tok/s:**

| Content type | Acceptance | c1 | c2 | c4 | **c6** |
|---|---|---|---|---|---|
| count (synthetic best case) | 5.96 | 84.6 | — | 216.1 | **304.5** |
| **code** (tool calls / structured / code) | **5.41** | 73.0 | 107.9 | 153.9 | **200.3** |
| **prose** (free-form reasoning) | **2.5** | 36.1 | 55.6 | 81.8 | **99.5** |

**Median single-stream within the same runs:**

| Content | c1 | c2 | c4 | c6 |
|---|---|---|---|---|
| code | 73.0 | 54.5 | 39.5 | **34.2** |
| prose | 36.1 | 28.1 | 20.5 | **17.3** |

Latency for a 600-token generation: code at c6, p50 **17.6s** / p95 **18.0s**.

**Diminishing returns; c6 is near saturation**: c1→c2 **+48%**, c2→c4 **+43%**, c4→c6 **+30%**.

> ### ⚠️ Do not benchmark c > 6
>
> `MAX_NUM_SEQS=6` is an engine hard ceiling (logs show `Running: 6, Waiting: 2`). Concurrency above 6 measures a **burst tail artifact** — the first six finish at c6 speed, the remainder then run alone at low concurrency, stretching wall-clock time and diluting the aggregate.
>
> The resulting number looks like "higher concurrency is slower", **but it is not engine capacity — it is an artifact of the test shape**. `loadtest.py` has a guard that refuses `c > 6`.

**Do not raise `MAX_NUM_SEQS`**: `Maximum concurrency` is only 2.44× (the KV pool holds roughly 2.4 full-length requests), so raising it makes long contexts trigger preemption sooner. **Preemption does not cost you a queue wait — it costs a full re-prefill of the entire prompt.**

### 5.4 Cross-validation against upstream

Upstream's capacity figures (553 requests / 213K tokens / 40 minutes) match this machine **item by item**:

| Upstream metric | Upstream | Measured here | Delta |
|---|---|---|---|
| c4 aggregate (BST prompt) | 151.1 | code c4 **153.9** | +1.9% |
| c4 single-stream (BST prompt) | 38.7 | code c4 **39.5** | +2.1% |
| c4 aggregate (real mixed traffic) | 88.6 | prose c4 **81.8** | −7.7% |
| c4 single-stream (real mixed traffic) | 22.3 | prose c4 **20.5** | −8.1% |

**Two conclusions:**

1. Upstream's "BST prompt" benchmark is **code-class content**, and their "real mixed traffic" lands near the **prose tier** — both frames of reference are now pinned down
2. **This machine performs identically to upstream at the same configuration** — no hidden loss

### 5.5 How to use these tables for capacity planning

> **Plan conservatively with the prose tier: roughly 100 aggregate / 17 single-stream at c6. The optimistic ceiling is the code tier's 200 / 34.**

In a real agent workload:

- **Tool-call arguments, structured output, code generation** → acceptance near the code tier (~5.4)
- **Natural-language replies, reasoning chains** → near the prose tier (~2.5)

Upstream's advice to "plan on roughly 88 aggregate / 22 single-stream at c4" **still holds** — it corresponds to the prose end.

### 5.6 Prefill ⚠️ not benchmarked on this machine

> **The figures in this section come from upstream's campaign, not from measurements here.** Every other performance number in this document was measured on this hardware; this section is the exception and is marked as such.

Upstream data: **prefill gets faster with depth** — 8K → 1,513, 32K → 2,284, 100K → **2,639** tok/s, varying no more than ±2% across 10 configurations. **Compute-bound, not tunable.**

**What was and was not measured here:**

- ❌ **No dedicated prefill benchmark was run.** All four `bench-baseline.py` prompts carry only ~28 input tokens, so prefill is negligible
- ⚠️ `loadtest.py`'s mixed mode does include a ~5000-token prompt, but the `input throughput` it reports is **`total input tokens ÷ total wall clock`**, and that denominator includes decode time — **it is not prefill throughput and under-reports badly**. It is only useful for comparing relative change across concurrency levels

#### How to measure it (scripts ready, results pending)

**Do not isolate prefill with `max_tokens: 1`** — that is a synthetic condition. With chunked prefill enabled, prefill already shares scheduling steps with decode, so an isolated figure runs optimistic and does not reflect real use.

Use two quantities that hold under real conditions instead. `scripts/measure-prefill.py` has a mode for each:

```bash
./scripts/measure-prefill.py ttft --md        # TTFT vs prompt length
./scripts/measure-prefill.py blocking --md    # how long a short request waits behind a long prefill
```

**① TTFT (`ttft` mode)** — the quantity users actually feel. Sweeps 1K / 8K / 32K / 100K, median of 3 per length. It also reads the engine's own `Avg prompt throughput` (vLLM already separates prefill from decode internally — no synthetic setup needed).

> Using **streaming** for TTFT is correct here and does not contradict §6.1's "throughput must use `stream:false`". They are different metrics: throughput counts tokens, and a streaming chunk holds all tokens accepted in one step; TTFT only cares when the first chunk arrives, which only streaming exposes.

**② How long a short request waits (`blocking` mode)** — measures short-request TTFT while idle as a baseline, then again with a long request's prefill in flight, and reports the ratio.

**This is the real question for a mixed workload**, and the only basis for judging whether §8.4's two prefill scheduling parameters are worth anything: that section derived `long_prefill_token_threshold`'s mechanism from source (a hard per-step token cap turning 8 steps into 59), but **the other side of the trade — how long short requests actually wait — was never measured**.

- Ratio near 1 → chunked prefill is already letting short requests interleave, and those parameters are unnecessary
- Ratio large → long prefill is monopolizing scheduling steps, and only then is the trade worth reconsidering

**Fill results back into this section**, recording the configuration quadruple and the date alongside (§6.5).

### 5.7 KV pool: expected values and noise

At the production configuration the KV pool runs about **1.24M–1.31M tokens**, with `Maximum concurrency` around 2.44–2.50×.

> ### ⚠️ KV pool varies up to 11% between boots — a single measurement cannot evaluate a config change
>
> Measured across multiple boots at identical configuration:
>
> | Configuration | KV pool across boots | Spread |
> |---|---|---|
> | gmu 0.78 / accounting off | 1,115,248 / 1,216,093 / 1,237,098 | **11%** |
> | gmu 0.7935 / accounting on / Patch A on | 1,242,845 / 1,278,794 | **2.9%** |
>
> Two conclusions were once drawn from single measurements and are now retracted: "Patch A costs 3.1% of KV" (the ON arm varies 2.9% by itself) and "gmu tuning gained 8%" (the real gain is about +3–4%).
>
> **The KV pool does not jump transiently the way throughput does, but it still requires multiple samples.**

---

## 6. Measurement discipline

Every rule here was written down after being violated. **Breaking them yields systematically wrong numbers, not merely noisy ones.**

### 6.1 `"stream": false` is mandatory

Under speculative decoding vLLM emits at most one SSE chunk per decode **step**, containing **all** tokens accepted in that step.

**Counting streaming deltas measures steps/sec, not tokens/sec** — the same request read 14.7 vs 60.1, a **4× under-report**.

Correct approach: `"stream": false`, read `usage.completion_tokens`.

### 6.2 Warm up to steady state

**Idle decay costs 30%, and the startup log gives no hint of it** — the server reports ready, cudagraphs are captured, answers are correct throughout, and it is simply slow.

Upstream measurements:

```
Just started (graphs captured, 3 short warmups sent)   58.5 tok/s
After roughly 5 long generations                       83.3 / 83.2 / 83.1 / 83.2
```

**And it decays again**: same container, no restart, 40 minutes of load followed by ~30 minutes idle put count300 at **60.4**; heavy re-warming restored **83.5**.

**A few 100-token warmups are not enough — you need 500–700-token generations.** `bench-baseline.py warmup` runs 5 rounds of ~500 tokens.

**Countermeasure: a keep-warm cron** (`scripts/keepalive.sh`, one 600-token generation every 15 minutes, ~0.8% duty cycle):

```
*/15 * * * * /home/<user>/services/dsv4f/keepalive.sh
```

Key points:

- Check `vllm:num_requests_running` and skip the timing run when non-zero (do not steal a concurrency slot) — **but still push the monitoring heartbeat**, or the system reports a fault exactly when it is busiest
- **Prefix the prompt with a nonce** to bypass prefix caching, so the prefill path stays warm and the numbers stay comparable
- Log both tok/s and acceptance length; the log then *is* the health curve

### 6.3 Load-test prompts need a nonce

Without one, the second run of the same long prompt hits the prefix cache and **wall-clock drops from 5.58s to 0.99s** — you are no longer measuring the prefill path.

**The nonce must be at the very front of the prompt** (prefix caching matches on prefixes).

### 6.4 Take the median of at least three runs

**This machine produces 2× transient outliers.** Measured: count-300 once reported **39.6 tok/s** against a normal 82.6; two re-runs returned 82.4 / 82.6.

Ruled out at the time: `Running: 1 / Waiting: 0` throughout, so **no competing traffic**; acceptance length 6.00 (perfect), so **draft quality was fine**. Not load, not speculative-decoding degradation — a transient **doubling of per-step time**.

Common factor: all such outliers appeared on **the first heavy request after an idle gap**.

> Noise floors measured on this machine:
>
> | Metric | Noise |
> |---|---|
> | Single-stream baseline (median of 3) | **±1%** |
> | Concurrency aggregate at c6 | **±11%**, plus occasional 2× outliers |
> | KV pool (boot to boot) | **±11%** |
>
> **When an effect is smaller than the noise floor, the experiment can only ever produce "no difference," no matter how many times you run it.** Know a metric's noise before choosing it.

### 6.5 Three things every reported number needs

1. **The configuration quadruple** — otherwise the reader does not know which configuration you measured
2. **Acceptance length** — otherwise they cannot tell a slow system from hard content
3. **Temperature** — the T>0 penalty is real: the drafter exports one-hot probabilities, so prose sits around 40% at t=0 but **only 27.6% at T=0.7**. Production agents usually run T>0

---

## 7. Failure modes

### 7.1 The core risk: runs fine, quality fine, half the speed

**The biggest risk in this deployment is not a crash.** Speculative decoding is verified by the target model, so a bad draft can only make you slower — **never wrong**.

So the typical failure looks like this: service starts normally, self-checks pass, answers are entirely correct, **throughput is halved**.

> **"Slower with no quality loss" is itself the diagnosis** — it points at the drafter, not at the weights or the configuration.

**Master gauge: mean acceptance length ≥ 3.5** (healthy 4.01–5.19, failed around 2.28).

Watch it live:

```bash
docker logs -f --tail=0 dsv4f-vllm-dspark-1 2>&1 \
  | grep --line-buffered -oE "Mean acceptance length: [0-9.]+|Avg generation throughput: [0-9.]+ tokens/s"
```

### 7.2 Four silent failure points

| # | Failure | Symptom | How to check |
|---|---|---|---|
| 1 | **Patch 3 not loaded** | 11/12 cold prefills fail, 0/19 hot requests fail | `docker exec <c> grep -c is_prefill_chunk …/sched/scheduler.py` → 5 |
| 2 | **Patch 4 not in effect** | Acceptance 60.2% → 25.7%, 55.4 → 32.7 tok/s | `docker exec <c> grep -c shared_experts.gate_up_proj …/spec_decode/dspark.py` → ≥2 |
| 3 | **B12X MoE off** | Silent fallback to DEEPGEMM_MXFP4, 55+ → 29 tok/s | `docker exec <c> env \| grep VLLM_USE_B12X_MOE` → 1 |
| 4 | **Patch 5 (stop strings)** | Client sends `stop` and gets `content: null` — the answer silently disappears | See [9.4](#94-prerequisites-for-enabling-thinking-mode) |

**Patch 3 deserves a separate note**: it **only manifests on cold prefill**; hot requests all pass, so smoke tests never catch it. Upstream measured k=3 failing 10/10 as well — **`k` is not the variable, Patch 3 is**.

### 7.3 Two traps in the checking method itself

These make the checks above return **stable, wrong answers** — worse than having no check.

#### Trap 1: `grep -q` inside a `pipefail` pipeline

```bash
set -uo pipefail
docker logs "$CONTAINER" 2>&1 | grep -q "pattern"
```

**`grep -q` exits the instant it matches and closes the pipe** → upstream `docker logs` is killed by SIGPIPE (exit **141**) → `pipefail` judges the whole pipeline **failed**, even though the match succeeded.

Measured: exit code 141 under `pipefail`, 0 without it.

**The result is two checks broken in opposite directions:**

| Check as written | Actual behavior |
|---|---|
| `grep -q "warning string" && bad \|\| ok` | Pipeline always fails → **always takes `ok`, always passes silently**. The check never existed |
| `grep -q "success string" && ok \|\| bad` | **Always takes `bad`**, regardless of whether the log is present |

**Fix** — use `grep -c`, which reads all input and produces no SIGPIPE. It returns 1 when the count is 0, so `|| true` is needed:

```bash
n=$(docker logs "$CONTAINER" 2>&1 | grep -c "pattern" || true)
[ "${n:-0}" -gt 0 ] && ok "..." || warn "..."
```

> **The larger the upstream output, the more certain the trigger.** `docker logs` (several MB) hits it 100% of the time; `echo "$var" | grep -q` fills the pipe buffer in one write and is effectively safe — but there is no reason to leave it.
>
> **When reviewing a self-check script, treat every `| grep -q` as a suspect.**

#### Trap 2: one-shot logs get overwritten by the ring buffer

Startup-time lines like `Using 'B12X' Mxfp4 MoE backend` become ungreppable after a few hours.

**This is a second, independent problem from trap 1** — even with SIGPIPE fixed, log-based checks still expire on a long-running service.

**So all three of the first silent-failure checks should read file contents or environment variables via `docker exec`, never logs.**

### 7.4 Network clamped at 13 Gb/s

**Symptom**: RDMA bandwidth pinned at 13.41 / 13.36 Gb/s against a nominal 200; 26 combined across both rails.

**Elimination path (all negative, but worth recording):**

| Hypothesis | Ruled out by |
|---|---|
| Cables crossed | ARP table confirms same-named port to same-named port |
| PCIe downtrain | `LnkSta: Speed 32GT/s, Width x4` — full rate |
| Congestion control / retransmits / pause frames | hw_counters and ethtool counters all zero |
| SMMU address translation | Hugepage test showed no improvement |
| Single QP / small-message overhead | **QP×8 and message×16, four combinations — numbers moved less than 5%** |

Latency `t_avg = 1.67 µs` (normal is 2–5). **Perfect latency + clamped bandwidth + total parameter insensitivity = a clamp, not an overhead problem.**

> **A performance problem that is completely insensitive to parameters is not a tuning problem.** Overhead problems always vary with parameters; clamps do not. When the sweep curve is flat, stop tuning and go look at state.

**Root cause**: a known GB10 **initialization / hot-plug state**. **The fix is a reboot** (see [3.1](#31-fix-the-network-before-anything-else)).

**Two diagnostic traps:**

- `Detected insufficient power on the PCIe slot (27W)` in `dmesg` is **decorative** — it keeps printing after the fix and cannot be used as a diagnosis
- The community attributed this fix to a driver upgrade (580.126 → 580.142), but this machine still failed on 580.173 — **what actually helped was the reboot that came with the upgrade**

**GB10 network topology** (4 interfaces = 2 physical ports):

| Interface | PCIe domain | RDMA device | Physical port |
|---|---|---|---|
| enp1s0f0np0 | 0000:01:00.0 | rocep1s0f0 | Port 0 |
| **enp1s0f1np1** | 0000:01:00.1 | **rocep1s0f1** | **Port 1** |
| enP2p1s0f0np0 | 0002:01:00.0 | roceP2p1s0f0 | Port 0 |
| **enP2p1s0f1np1** | 0002:01:00.1 | **roceP2p1s0f1** | **Port 1** |

> **Note**: upstream measured the performance impact of dual HCA as **NULL** — the interconnect is not this workload's bottleneck (per-step collectives are latency-bound; prefill is compute-bound on GB10). Fixing 13 Gb/s took it from "broken" to "adequate", not from adequate to better.

### 7.5 High KV cache usage is not a problem

With `--enable-prefix-caching` on, KV blocks from completed requests stay in the pool as cache until space is needed.

**95% usage is normal, and arguably means the cache is working.**

**The real pressure signals are non-zero `Waiting` and preemption:**

```bash
docker logs --tail=300 dsv4f-vllm-dspark-1 2>&1 | grep -E "Waiting: [1-9]|Preempt"
```

Preemption does not cost a queue wait — it costs **a full re-prefill of the entire prompt**.

---

## 8. Tuning record: what was tried, what the verdict was

### 8.1 Summary

| Change | Verdict | What it actually bought |
|---|---|---|
| **Patch A** + `DRAFT_CAPTURE_SIZES=auto` | ✅ keep | **+6% single-stream**, the only solid performance gain |
| `ESTIMATE_CUDAGRAPHS=1` + gmu 0.7935 | ✅ keep | **Correctness** (CUDA-graph memory counted in allocation, less edge OOM), KV +3–4% |
| `FUSED_MARKOV_ARGMAX=1` | ✅ keep | Neutral. Kept because it should save memory bandwidth in theory and measurably does no harm — **not because it measured faster** |
| Two prefill scheduling params | ❌ disproven | One is dead code on V1; the other's behavior is the opposite of its name |
| Narrowing cudagraph capture sizes | ❌ disproven | Completely neutral; both benefit and cost sit below the noise floor |

**Of four changes, only Patch A actually delivered performance.** The other two bought correctness and theoretical headroom.

### 8.2 gmu and cudagraph accounting must change together

**The gmu value suggested by the startup log depends on whether accounting was on or off when it printed** — the two are not the same arithmetic:

| Configuration at the time | Log says "equivalent to" | Log suggests |
|---|---|---|
| 0.7800 / accounting off | — | 0.7873 |
| 0.7873 / accounting on | 0.7811 | 0.7935 |
| 0.7935 / accounting on | 0.7860 | 0.8010 |

**Following the accounting-off suggestion (0.7873) while turning accounting on shrinks the KV pool by 5.4%.**

Correct procedure: **turn accounting on, boot once, then follow the newly printed suggestion.**

The offset is roughly **0.006–0.0075**: subtract it from an accounting-on gmu to get the accounting-off equivalent. So **0.7935 (on) ≈ 0.7860 (off)**, still below the 0.80 physical edge found by upstream's campaign.

> **An automatic suggestion is not a constant — it is a function of current state. Change the state and re-read the suggestion.**

### 8.3 Patch A: the only solid gain

**The decisive evidence is the single-stream baseline** (the concurrency sweep is too noisy and hid the real effect). Canonical prompts, three runs each, spread ≈1%:

| Test | `off` | `auto` | Delta |
|---|---|---|---|
| count | 81.0 | **85.7** | **+5.8%** |
| struct | 70.7 | **75.0** | **+6.1%** |

Upstream reports +5% at c4; measured here c4 came in at +2.3–6.2%, **the same order of magnitude**.

> **Lesson**: an earlier evaluation used the concurrency sweep (±11%, occasional 2× outliers) and concluded "neutral to mixed", nearly abandoning the patch. **A noisy metric applied to a small effect produces a false negative.**

**One thing in the experimental design is worth copying**: the patch states `off -> byte-identical legacy behaviour`. That means **you never need to rebuild an image for the control arm** — same image, same code, flip one environment variable, and you have a textbook A/B.

**And that promise is itself worth verifying**: setting `off` removed the activation log and returned the KV pool to its prior value, confirming the off path is clean. Had the numbers **not** come back, that would have been the more important finding — it would have meant the whole controlled comparison rested on a false premise.

**Side benefit**: when the target's capture sizes were cut from 12 buckets to 7, the drafter's `[1,2,4,6]` did not move and the log correctly reported `target sizes stay [...]`. `_DrafterCompilationConfigView` does exactly what it claims.

### 8.4 Disproven: the two prefill scheduling parameters

```
--long-prefill-token-threshold 1024
--max-num-partial-prefills 1
```

Copied from upstream's sparkrun port, understood as "limit long prefills to one at a time". **Reading the source shows neither claim holds:**

**① `max_num_partial_prefills` is dead code on V1.**
`grep -rn max_num_partial_prefills /opt/env/…/vllm/v1/` returns **zero hits** — it exists only on the V0 path, and this deployment runs V1 (startup log: `Initializing a V1 LLM engine`). Its default is 1 anyway, so it is doubly inert.

**② `long_prefill_token_threshold` is not a concurrency limit — it is a hard per-request per-step token cap.**

```python
num_new_tokens = request.num_tokens - num_computed_tokens
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

Both call sites share this logic, and **neither is gated on `max_num_partial_prefills`**.

What setting it to 1024 actually does: a 60K-token prompt goes from 8192 tokens per step (capped by `max_num_batched_tokens`) down to 1024 — **from about 8 steps to about 59**.

It does achieve the goal of keeping short interactions from being blocked behind a long prefill — each step consumes only 1024 of the budget, leaving 7168 for others. **But the price is giving up "prefill is compute-bound and gets faster with depth" (8K→1,513, 100K→2,639 tok/s).**

**Verdict: a latency-for-throughput trade, not a free optimization.**

> **A parameter's name describes intent, not behavior.** `long_prefill_token_threshold` sounds like "the threshold for classifying a prefill as long"; reading it that way produces exactly the opposite conclusion about its value. **Before deciding whether to use a parameter, find the line that reads it.**

### 8.5 Disproven: narrowing cudagraph capture sizes

**The motivation looked solid**: vLLM defaults to

```
max_cudagraph_capture_size = min(max_num_seqs × (1+k) × 2, 512) = min(6×6×2, 512) = 72
```

where `×2` is generic headroom. But `--max-num-seqs 6` hard-caps concurrent sequences (load-test logs confirm `Running` never exceeded 6), so **the ceiling for a pure decode batch is 6×6 = 36** and sizes 40/48/56/64/72 can never be used.

Measured with `--max-cudagraph-capture-size 36`:

| Metric | Default 72 | Narrowed to 36 |
|---|---|---|
| Capture list | `[1,2,4,8,16,24,32,40,48,56,64,72]` | `[1,2,4,8,16,24,32]` |
| Graph capturing | 0.74 GiB / 9s | **0.40 GiB / 4s** |
| KV pool | 1,278,794 | 1,279,421 (**unchanged**) |

**Verdict: completely neutral — both benefit and cost sit below the noise floor.**

**Why the 0.34 GiB saved never became KV**: the expected gain is ≈2.6%, while the KV pool's boot-to-boot variance is 2.9% ([5.7](#57-kv-pool-expected-values-and-noise)). The effect is smaller than the noise.

**The value itself was also chosen wrong**: vLLM generates buckets `[…24, 32, 40…]`, so capping at 36 stops the list at **32** while a full c6 batch is **36** — it falls between two buckets, no graph covers it, and it drops to eager.

> **Before setting a "maximum" parameter, confirm the value lands on a bucket the system actually generates.** Maximum parameters usually truncate a discrete sequence rather than being continuous. Covering 36 requires setting **40**.

If revisited, set 40 rather than 36 — that only removes the definitively unusable 48/56/64/72. But the benefit still sits below the noise floor: **what cannot be measured does not exist, and is not worth carrying an unproven divergence for**.

`docker-compose.dspark.yml` keeps the parameter in optional form; leaving the variable empty passes nothing:

```yaml
${MAX_CUDAGRAPH_CAPTURE_SIZE:+--max-cudagraph-capture-size ${MAX_CUDAGRAPH_CAPTURE_SIZE}}
```

### 8.6 Explicitly not doing

| Item | Reason |
|---|---|
| k = 4 or 3 | Three independent lines of evidence pin k=5 ([4.3](#43-k5-is-a-hard-constraint)) |
| gmu at 0.80 / 0.85 | Campaign found 0.80 already at the physical edge. **Mind the frame**: 0.7935 is the accounting-**on** value, equivalent to 0.7860 off |
| The two prefill scheduling params | Disproven ([8.4](#84-disproven-the-two-prefill-scheduling-parameters)) |
| Narrowing cudagraph capture sizes | Disproven ([8.5](#85-disproven-narrowing-cudagraph-capture-sizes)) |
| Going to 1M context | The client side is 524288 too, so raising the server gains nothing; sparse MLA also allocates a max-length-dependent workspace |
| `VLLM_USE_B12X_MHC=1` | Campaign E9: crashes at startup (`Can't export tensors that require gradient`) |
| `VLLM_DSV4_B12X_COMPRESSED_MLA=1` | Campaign E12: **wrong output** plus CUDA assert |
| `VLLM_USE_B12X_FP8_GEMM=1` | Upstream docs warn it hits a DeepGEMM layout assertion during DSpark drafter warmup |
| Dual-HCA tuning | Campaign E6 measured null; the interconnect is not the bottleneck |
| `--max-num-batched-tokens 16384` | Campaign E7 hit a KV cliff |
| The three MLA sparse sub-parameters | Prefill is compute-bound; differences fall inside ±2% noise |
| `--block-size` 128 or 512 | Kernels require exactly 256 |

> **The B12X family has a precedent for running fine while computing wrong results** (`COMPRESSED_MLA`). Any switch in that family must be validated on output correctness, not just speed.

### 8.7 An untested candidate

`VLLM_USE_B12X_SPARSE_INDEXER=1` — the only B12X switch with neither a documented warning nor upstream campaign coverage. All gates pass: `is_device_capability_family(120)` returns True for SM121, and `use_fp4_indexer_cache` defaults to False.

**Validation must compare output correctness.**

---

## 9. Client integration notes

This section covers **the server API's own behavior** — anyone integrating a client hits these, regardless of which framework they use.

### 9.1 The response field is `reasoning`, not `reasoning_content`

**This runtime has no `reasoning_content` key** — it is deprecated and accepted only on *input*.

A client reading `reasoning_content` sees an empty value forever, then concludes "reasoning extraction is broken". The same applies while streaming: the delta carries `delta.reasoning`.

### 9.2 `reasoning_effort` genuinely enables thinking mode

Measured (`stream:false`, direct to the server):

| Request | `reasoning` field | completion_tokens |
|---|---|---|
| No `reasoning_effort` | **empty** | 224 |
| `reasoning_effort: "medium"` | **populated** (structured CoT) | **398** |
| `reasoning_effort: "high"` | **populated** (terse CoT) | 77 |

**It is equivalent to `chat_template_kwargs: {"thinking": true}`.**

**You pay twice**: the same question costs 398 tokens with `medium` versus 224 without, and the extra tokens are prose-class chain-of-thought — where acceptance runs only 25–33%, so **those tokens are also unusually slow**.

> This was once recorded as "setting high probably does nothing" — **exactly backwards**. That test only compared `content`, which is genuinely clean on both sides of the switch. **All of the change happened in the `reasoning` field, which was reading empty because of the field-name issue in 9.1.**
>
> **Before judging whether a switch works, confirm you are looking at the field it actually changes.**

### 9.3 The server parser is fine — leaks are client-side

Four combinations measured, and **CoT never entered `content`**:

| stream | thinking | `reasoning` field | `content` |
|---|---|---|---|
| false | false | empty | clean |
| false | true | full CoT | clean |
| true | false | empty | clean |
| true | true | 136 deltas | clean |

**So if chain-of-thought shows up in your client's replies, the server is not the cause.** Check in this order:

1. **Source** — does the outgoing request carry `reasoning_effort`? (Parse the JSON and inspect `request.body` keys; **do not grep the file** — system prompts may contain explanatory text mentioning it, and grep will false-positive)
2. **Middle layer** — if a proxy or gateway sits between client and server, check whether it has an option to merge reasoning into content. **Such a switch produces a symptom indistinguishable from a server-side leak while the server is entirely innocent** (LiteLLM's `merge_reasoning_content_in_choices`, for example)
3. **The client's display switches** — a framework may have several independent switches controlling chain-of-thought display, where changing one does not affect the others. It may also have a "config exists but code never reads it" failure, where the setting is written at one nesting level while the loader reads another. Failures of that kind raise no error and can only be found by reading the source
4. **Only then suspect the server parser**

**Turning it off at the source is far cleaner than guessing which display switch matters.**

### 9.4 Prerequisites for enabling thinking mode

Two equivalent routes: `chat_template_kwargs: {"thinking":true}` or `reasoning_effort` — either suffices. Before enabling, confirm:

1. **The client reads the `reasoning` field** (9.1)
2. **Patch 5 becomes mandatory** — in thinking mode generation starts *inside* `<think>`, the chain-of-thought restates prompt phrases, the client's `stop` fires early, `</think>` never arrives, and you get **`content: null` — the answer silently disappears**
3. **Speed drops to the prose tier** (about 35–40 tok/s) and token consumption rises sharply (224 → 398)
4. **Both the middle layer and the client's display switches need checking** (9.3)

---

## 10. Methodology

These 26 items were each paid for during this deployment. The principles referenced throughout the document are collected here.

### On measurement

**1. A performance problem that is completely insensitive to parameters is not a tuning problem.**
QP×8, message×16, hugepages, jumbo frames — four variables swept, numbers moved less than 5%. Overhead problems always vary with parameters; clamps do not. **When the sweep curve is flat, stop tuning and go look at state.**

**2. Define the noise floor before deciding what counts.**
Re-run the baseline unchanged once to establish noise; any result must clear the floor to count. Otherwise you will bank noise as gains.

**3. This machine produces 2× transient outliers — a single measurement is never a conclusion.**
count-300 once read 39.6 against a normal 82.6; two re-runs returned 82.4 / 82.6, with no competing traffic and perfect acceptance length. **This nearly caused a harmless change to be misjudged as a −52% regression and rolled back.**

**4. A noisy metric applied to a small effect produces a false negative.**
The concurrency sweep (±11%, occasional 2× outliers) evaluated Patch A as "neutral to mixed" and nearly caused it to be abandoned. The single-stream baseline (±1%) resolved a solid +6%. **When the effect is smaller than the floor, no number of repetitions will ever show a difference.**

**5. A throughput number without an acceptance length is uninterpretable.**
Same machine, same boot: single-stream from 36 to 85 tok/s, **the entire 2.3× spread coming from content acceptance rate**, while GPU step rate stays essentially constant.

**6. Change the prompt and it is no longer the same baseline.**
For trend comparison the **prompts must be frozen verbatim and committed**, or every "baseline" is a fresh one.

**7. "Doesn't match the historical number" has two explanations, and you cannot separate them without re-measuring the control.**
A baseline was 11% below its historical value and looked like a regression. Reverting every variable and re-measuring produced **the same number at the original configuration** — the historical value was the irreproducible one. And the changes had actually improved it by 7.3%.

**8. Nominal and achievable are different things.**
A nominally 200G port on GB10 caps near 126 Gb/s per PCIe domain, with a measured ceiling of 196.

**9. When comparing recipes, first ask what each one optimizes.**
Four nodes without speculation at 256-way concurrency versus two nodes at k=5 single-stream — comparing absolute numbers is meaningless.

### On configuration and parameters

**10. Before copying a parameter, confirm it is still alive on your code path.**
`max_num_partial_prefills` has zero hits in the V1 code — it is a V0 leftover. The parameter is accepted, raises no error, appears on the command line, renders correctly in `docker compose config`, **and the engine never reads it**. This class of failure is silent and only source reading finds it.

**11. A parameter's name describes intent, not behavior.**
`long_prefill_token_threshold` sounds like a classification threshold; it is a hard per-request per-step token cap. **Before deciding to use a parameter, find the line that reads it.**

**12. Before setting a "maximum" parameter, confirm the value lands on a bucket the system generates.**
Setting `--max-cudagraph-capture-size 36` from a real ceiling of 36 stopped the generated list at 32 (buckets are `[…24, 32, 40…]`), and a full batch lost its cudagraph entirely. **Maximum parameters usually truncate a discrete sequence.**

**13. When software suggests a value, ask what state it computed that in.**
The startup log suggested raising gmu to 0.7873; following it **shrank the KV pool by 5.4%**, because that suggestion was printed with accounting off while accounting was being turned on. **An automatic suggestion is a function of current state, not a constant.**

**14. A template's defaults may lag the measured results in the same repository.**
`.env.example` says `gmu=0.85` while the campaign five days earlier concluded 0.78; the sparkrun port says `k=3` while the README states that is a bug value. **Campaign results went into the docs and were never synced back into the template.**

**15. Check which version a community number was measured on.**
Someone attributed the 13 Gb/s fix to a driver upgrade, but this machine still failed on a newer driver — **what actually worked was the reboot that came with it**.

### On checks and monitoring

**16. For every monitoring signal, ask when it lies.**
The keep-warm script skips its timing run when busy → skips the heartbeat → reports a fault exactly when the system is busiest. **Monitoring with blind spots is more dangerous than no monitoring** — it creates the illusion of coverage.

**17. `grep -q` inside a `pipefail` pipeline turns "matched" into "exit failure".**
`grep -q` exits on match and closes the pipe; upstream dies of SIGPIPE (141) and `pipefail` fails the pipeline. **Such a check raises no error and returns a stable, wrong answer** — always passing or always failing depending on whether you hung `&&` or `||` off it. **Treat every `| grep -q` in a self-check script as a suspect.**

**18. One-shot logs cannot serve as a long-term criterion.**
The ring buffer overwrites startup logs. To check whether configuration took effect, read files or environment variables with `docker exec`.

**19. Grepping a string is not the same as a parameter taking effect.**
`reasoning_effort` is greppable in session dumps, but that text lives in a system prompt. **Substring matching in structured data tells you it was mentioned, not that it was used.**

**20. Comparing only the visible field yields the opposite conclusion.**
`reasoning_effort` was judged inert because the comparison looked at `content`; all of the difference was in the `reasoning` field. **Before judging whether a switch works, confirm you are looking at the field it actually changes.**

### On change and collaboration

**21. Before installing a patch, look for the repository's own patch mechanism.**
Using the built-in mechanism gives you cross-node consistency, staleness checking, and post-build smoke tests for free. **A hand-rolled mount raises no error — it just becomes an undiagnosable bug several months later.**

**22. A feature switch that claims "off is equivalent" is a free control arm.**
Same image, same code, one environment variable flipped — a textbook A/B. **And that promise is itself worth verifying**: if the numbers do not come back when you turn it off, the entire controlled comparison rested on a false premise.

**23. A missing layer in the topology throws off the whole diagnostic chain.**
The middle proxy carries switches that rewrite requests and responses. **Diagnosing against a wrong topology attributes the middle layer's behavior to the endpoints.**

**24. Writing config files with a heredoc can silently produce half a file.**
`EOF` becoming content, variables left unexpanded — neither raises an error. Splitting into `cp` plus several `sed` calls avoids this: each command either succeeds or fails visibly, and `diff` tells you exactly what changed.

**25. The verification must cover the kind of change made — a syntax check catches no semantic error.**
After a bulk regex edit to a script, only a syntax check was run. The result: imports removed while an old function still referenced those names, so it raised **NameError on the first run** despite being perfectly valid syntax. The same edit also left the documentation describing a behavior the code did not have ("empty reference means no deviation column" — it actually raised KeyError).
**If you change code, run the code.** When the service is unavailable, at minimum confirm it fails at the real work (the network call) rather than at import or name resolution. And **"reaches the network call" still is not "completes"** — the entire response-parsing path lies between them.

**26. Before troubleshooting "flaky", count how many instances are actually running.**
An old client instance once ran for nearly five days unnoticed, its configuration frozen at a pre-migration state and requesting a model name the server answered with 404 — it caused no visible failure purely by luck. **A second instance of the same service is the hardest variable to think of**, because everyone's mental model contains only one.

---

## 11. References

- **Upstream recipe**: [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) (MIT)
  - `RESULTS-BLUEY-2026-08-20.md` — the 16-boot tuning campaign; the "campaign E6/E7/E9/E12" references throughout this document come from there
  - `AGENT_GARBLE_FIX.md` — Patch 3 and cold-start garbling
  - `DSPARK-SHARED-EXPERT-FIX.md` — Patch 4
- **DSpark concurrency patch**: Keys / drowzeys, see [CREDITS.md](../CREDITS.md)
- NVIDIA developer forum thread 363461 — **the actual fix for 13 Gb/s (reboot)**
- Chronara, "GX10 ConnectX-7: Why You're Getting 13 Gbps" — GB10 PCIe topology
- vLLM issues #51318 / #52836 / #52492 — garbling

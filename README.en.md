[中文](README.md) · **English**

# DeepSeek-V4-Flash-0731 on Two GX10 Boxes

DeepSeek-V4-Flash-0731 running on two NVIDIA GB10 (DGX Spark / GX10) nodes: TP=2, DSpark speculative decoding at k=5, NVFP4 KV cache.

This repo is a **deployment record**, not a tutorial: configuration, self-checks, measurement tooling, and a set of conclusions that measurement later overturned. Every number here was measured on this hardware.

> ## ⚠️ This measures the hardware's limits, not the model's quality
>
> **There is no capability evaluation anywhere in this repo** — no benchmark scores, no output-quality comparisons, no conclusion about whether this model is any good.
>
> Everything here answers one question: **how much this hardware plus this configuration can sustain, where it tops out, and what makes it silently run at half speed.**
>
> That framing explains several choices that would otherwise look odd. The four standard test prompts (count / struct / code / prose) were picked to span **acceptance-rate tiers**, not capability dimensions — their job is to cover the range from "highly predictable" to "free-form text" so that throughput numbers become interpretable (see [the throughput model](docs/deployment-notes.en.md#2-read-this-first-the-throughput-model)). **Using them to judge model capability would be meaningless.**
>
> Likewise, the noise floors, medians-of-three, and silent-failure hunting that recur throughout exist to make **the hardware-side numbers trustworthy** — they say nothing about output quality.

**Two things to know before reading:**

- This configuration optimizes **single-stream latency** (agent workloads), not high-concurrency throughput. Comparing its absolute numbers against a four-node non-speculative throughput setup is meaningless.
- The biggest risk in this deployment is not a crash. It is **"it runs, output quality is fine, and it's half as fast."** A large part of this documentation is about catching that class of silent failure.

---

## Production configuration

```
gmu 0.7935 / cudagraph accounting 1 / fused-markov 1 / DRAFT_CAPTURE_SIZES auto
```

| | |
|---|---|
| Endpoint | `http://<head>:8078/v1`, model name `deepseek-v4-flash` |
| Hardware | 2 × NVIDIA GB10 (SM121), 128 GB unified memory, CX7 dual-rail RoCE |
| vLLM | `0.21.1rc1.dev339+g1967a5627bc3` |
| Weights | `deepseek-ai/DeepSeek-V4-Flash-0731`, 155.43 GiB / 48 shards |
| Image | `vllm-dspark-runtime:dspark-nvfp4-stage-c` — **must include Patch A** |

Verify everything with one command:

```bash
./scripts/dsv4f-launch.sh --check-only
```

## Performance baseline (2026-08-30)

Canonical prompts, after 5×500-token warmup, `stream:false`, t=0, median of 3 runs.

| Test | Single-stream tok/s | Acceptance length |
|---|---|---|
| count | 85.5 | 5.96 |
| struct | 74.6 | — |
| code | 71.6 | 5.41 |
| prose | 36.8 | 2.5 |

Aggregate throughput under concurrency (`MAX_NUM_SEQS=6` is a hard ceiling):

| Content | c1 | c2 | c4 | c6 |
|---|---|---|---|---|
| code | 73.0 | 107.9 | 153.9 | **200.3** |
| prose | 36.1 | 55.6 | 81.8 | **99.5** |

> **Always report acceptance length alongside throughput.** On this machine single-stream ranges from 36 to 85 tok/s — a 2.3× spread that comes entirely from how predictable the content is. GPU step rate stays roughly constant at ~14.5 steps/sec.
>
> To size a new workload: **single-stream tok/s ≈ 14.5 × acceptance length**, and c6 aggregate ≈ single-stream × 2.7.

## Prerequisites

| | |
|---|---|
| Hardware | **Two** NVIDIA GB10 nodes, 128 GB unified memory, CX7 dual-rail RoCE interconnect |
| OS | Ubuntu 24.04 aarch64, driver 580.x, CUDA 13.0 |
| Software | Docker + Compose, `bash`, `python3` (scripts use the standard library only — no pip install) |
| Network | **RDMA bandwidth must pass acceptance at ≥184 Gb/s first** — see notes §3.1. Do not skip this |
| Credentials | HuggingFace token (to pull 155.43 GiB of weights) |
| Upstream | This repo is a downstream record of the [upstream recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark); you need that build environment first |

## Layout

```
docs/deployment-notes.md   Full deployment notes (config, troubleshooting, capacity, 26 lessons)
scripts/
  dsv4f-launch.sh          Preflight → launch → six self-checks → optional benchmark
  bench-baseline.py        Single-stream baseline; prompts frozen so runs stay comparable
  loadtest.py              Concurrency sweep c1/2/4/6, nonce-prefixed to defeat prefix caching
  keepalive.sh             Warm-keeping cron, counters idle decay
  measure-prefill.py       TTFT vs prompt length; long-prefill blocking of short requests
  _common.py               Shared config and helpers (retarget the tools here)
  profiles/<name>.json     Per-model: endpoint, concurrency ceiling, speculative, baselines
patches/
  dspark_proposer.py       Patch A (drafter-private cudagraph sizes, Apache-2.0)
  README.md                Install instructions and measured results
config/
  env.dspark.example       Production config template (token stripped)
TODO.md                    Outstanding measurements and evidence-strength ledger
CHANGELOG.md               Configuration and tooling changes
NOTICE                     Per-file license attribution
```

### ⚠️ This repo is not the deployment directory

**This repo holds records and tooling, not a directory you can start the service from.** The actual deployment lives elsewhere (a checkout of the upstream recipe) and contains `start-deepseek-v4-flash-dspark.sh`, `.env.dspark`, `recipe/overlay/` and other things not included here.

`dsv4f-launch.sh` changes into the deployment directory to invoke those, with the path taken from `REPO_DIR`:

```bash
# assumes the deployment is at ~/services/dsv4f
./scripts/dsv4f-launch.sh --check-only

# deployment elsewhere
REPO_DIR=/opt/dsv4f ./scripts/dsv4f-launch.sh --check-only
```

Environment variables for `dsv4f-launch.sh` and `keepalive.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `REPO_DIR` | `~/services/dsv4f` | **The deployment directory** (not this repo) |
| `WORKER` | `192.168.100.2` | Worker's RoCE address; self-checks ssh there |
| `CONTAINER` | `dsv4f-vllm-dspark-1` | Container name |
| `PORT` / `MODEL` | `8078` / `deepseek-v4-flash` | Endpoint and model name |

> **The network values `HCA_A` / `IP_A` / `HCA_B` / `IP_B` are hardcoded in the script**, being tied to this deployment's dual-rail layout. A different network layout requires editing those lines directly.

### Using these tools on a different model

The measurement scripts contain no hardcoded deployment values; configuration
comes from a profile file or the environment:

```bash
# with an existing profile
PROFILE=dsv4f ./scripts/bench-baseline.py bench

# or straight from the environment (non-speculative model, ceiling of 16)
ENDPOINT=http://host:8000 MODEL=qwen3-32b MAX_NUM_SEQS=16 SPECULATIVE=0   ./scripts/loadtest.py sweep
```

For a new model, copy `scripts/profiles/dsv4f.json` and edit the values. The repo also ships `glm53.json` as a contrast — the same tooling pointed at a different model (different recipe, concurrency ceiling, and speculative mechanism), which shows which fields actually have to change.
Configurable: endpoint, model name, container name (for reading engine logs),
`max_num_seqs` (the concurrency guard), `speculative` (non-speculative models
skip acceptance-length collection), and reference baselines (omit them and the
deviation column is simply not printed).

Add `--md` to emit markdown tables ready to paste into the notes.

> ### ⚠️ These four Python tools have not been validated against a live service
>
> They share the request and parsing path in `_common.py`, and that code has only had syntax, config-loading, and concurrency-guard testing — **none has completed a real request**. Before relying on them, see [TODO item 0](TODO.en.md#0-validate-the-tooling-first--everything-else-depends-on-it), which gives a suggested validation order and lists exactly what is unverified.

## Getting started

1. Copy `config/env.dspark.example` to `.env.dspark` in your deployment directory; fill in your `HF_TOKEN` and both nodes' IPs
2. Copy `patches/dspark_proposer.py` over `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py` — the launch script rebuilds the image on both nodes automatically
3. **Set `VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto` in `.env.dspark`** — without it the patch is byte-identical to stock, i.e. installing it did nothing. See [patches/README.md](patches/README.en.md)
4. `./scripts/dsv4f-launch.sh` — deployment succeeded only when all six self-checks are green

## Three traps worth knowing up front

**1. Silent failure is more dangerous than a crash.** The risk is not that it won't start — it's "runs fine, quality is fine, half the speed." Speculative decoding is verified by the target model, so a bad drafter can only make you **slower, never wrong**. That means **"slower but no quality loss" is itself the diagnosis** — it points at the drafter, not the weights or config. The master gauge is mean acceptance length ≥ 3.5.

**2. `grep -q` inside a `pipefail` pipeline lies.** `grep -q` exits the moment it matches and closes the pipe; upstream `docker logs` is killed by SIGPIPE (exit 141), and `set -o pipefail` then judges the whole pipeline failed. **This makes a check return a stable, wrong answer** — one that always passes, another that always fails, depending on whether you hung `&&` or `||` off it. Use `grep -c` instead.

**3. A single measurement is never a conclusion.** This machine produces **2× transient outliers** (count-300 once read 39.6 tok/s against a normal 82.6; two re-runs returned 82.4 / 82.6). KV pool size varies up to 11% between boots. Take the median of at least three runs.

All 26 lessons are in the [deployment notes](docs/deployment-notes.en.md#10-methodology).

## Conclusions that measurement overturned

Much of this record's value is in **contradicting itself**. The main ones:

| What was believed | What measurement showed |
|---|---|
| The client talks to the server directly | A proxy sits in between, carrying switches that rewrite requests and responses |
| `reasoning_effort` does nothing | The opposite — it *is* the thinking-mode switch; token spend went 224 → 398 |
| The B12X false alarm was log rotation | The real cause was `grep -q` + `pipefail` (SIGPIPE), unrelated to whether the log is still there |
| A baseline had regressed 11% | Not a regression — the historical reference value is simply not reproducible; tuning actually improved it 7.3% |
| Two prefill scheduling params were worth copying | One is dead code on V1; the other's behavior is the opposite of its name |
| Narrowing cudagraph capture sizes saves memory | Completely neutral — both the benefit and the cost sit below the noise floor |
| Differing image IDs across nodes need fixing | A non-issue — the criterion is the md5 of files *inside* the image, not the image ID |

## License and attribution

**Licensing is not uniform across this repo. See [NOTICE](NOTICE) for the per-file breakdown.** Summary:

| Content | License |
|---|---|
| Notes, `bench-baseline.py`, `loadtest.py`, `patches/README.md` | MIT, original to this repo |
| `dsv4f-launch.sh`, `keepalive.sh`, `env.dspark.example` | MIT, derived from the [upstream recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) |
| **`patches/dspark_proposer.py`** | **Apache-2.0** — vLLM-derived; the original SPDX header is preserved |

Model weights, base images, CUDA / NCCL / FlashInfer / TileLang / Triton are not distributed here and carry their own terms.

**This repo is a downstream deployment record of the upstream recipe, not a replacement for it.** The DSpark concurrency work comes from Keys / drowzeys. Full attribution in [CREDITS.md](CREDITS.md).

[中文](README.md) · **English**

# Patch A — drafter-private cudagraph sizes

`dspark_proposer.py` replaces vLLM's `v1/spec_decode/dspark_proposer.py`. It is a clean superset of the stock file: +133 lines, −2.

> **License**: this file is **Apache-2.0**, derived from the vLLM project. The original SPDX header is preserved. It differs from the rest of this repository, which is MIT. See [NOTICE](../NOTICE).

## What it does

Gives the DSpark drafter a set of cudagraph capture sizes **independent of the target model**, keyed by **request count** rather than target token count. When enabled the log prints:

```
DSpark drafter-private cudagraph capture sizes enabled: [1, 2, 4, 6]
(target sizes stay [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72]).
Draft graphs are keyed by request count; a batch of B requests now runs B x 5 draft tokens.
```

## Measured benefit

Single-stream baseline, canonical prompts, three runs each (spread ≈1%):

| Test | `off` | `auto` | Delta |
|---|---|---|---|
| count | 81.0 | **85.7** | **+5.8%** |
| struct | 70.7 | **75.0** | **+6.1%** |

**This is the single most solid gain in the whole tuning round.** Upstream reports +5% at c4; measured here c4 came in at +2.3–6.2%, same order of magnitude.

> An earlier attempt evaluated this patch with a concurrency sweep and concluded "neutral to mixed" — nearly abandoning it. **That metric carries ±11% noise plus occasional 2× outliers, which swamped a real +6%.** A noisy metric applied to a small effect produces a false negative. The single-stream baseline (±1%) resolves it cleanly.

## Installing

**Do not use a bind-mount.** The upstream launch script says so directly:

> DSpark source patches ship inside the runtime image (`recipe/overlay/`), **not as runtime bind-mounts**.

The right move is to overwrite the overlay source and let the build chain rebuild:

```bash
cp patches/dspark_proposer.py \
   ~/services/dsv4f/recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py

cd ~/services/dsv4f && ./dsv4f-launch.sh
```

The launch script's overlay staleness check fires automatically:

```
verify-overlay-sources.sh → overlay image → stage-a → b → c
        ↓
build-dspark-vllm-runtime.sh defaults to WORKER_BUILD=1:
  rsync -az --delete the whole repo to the worker → worker rebuilds too
```

**Node consistency is guaranteed by design; no manual sync needed.** The whole chain takes about a minute per node.

## ⚠️ Installing the patch is not enough — you must also set the switch

```bash
VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto
```

**Without this variable the patch does nothing.** Its own parser says:

```
""/0/off  ->  None (feature off, byte-identical legacy behaviour)
```

The variable is also absent from upstream's `docker-compose.dspark.yml` `environment:` block, so writing it only into `.env.dspark` never reaches the container. Add:

```yaml
VLLM_DSPARK_DRAFT_CAPTURE_SIZES: "${VLLM_DSPARK_DRAFT_CAPTURE_SIZES:-}"
```

Empty means off, which is a safe default.

## Verifying

`dsv4f-launch.sh --check-only` checks two things:

- **The md5 of `dspark_proposer.py` inside the image matches across both nodes** — this is the direct guard against an NCCL hang. The criterion is **the file's md5, not the image ID**: each node runs its own `docker build`, so image IDs are inherently irreproducible
- **When set to `auto`, the activation log line must be present**, otherwise it errors

## Side benefit: the isolation is clean

When the target's capture sizes were cut from 12 buckets down to 7, the drafter's `[1,2,4,6]` did not move, and the log correctly reported `target sizes stay [...]`. `_DrafterCompilationConfigView` does exactly what it claims.

The "off means byte-identical" promise was verified too: setting `off` removes the activation log and returns the KV pool to its prior value. **That means you can run a clean A/B on a single image without rebuilding for the control arm** — flip one environment variable and nothing else changes.

## Attribution

The patch comes from the [upstream recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark). The DSpark concurrency work is by Keys / drowzeys — see [CREDITS.md](../CREDITS.md).

[中文](CHANGELOG.md) · **English**

# Changelog

Changes to the deployment configuration and tooling. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Dates are measurement dates. How every performance number was measured is documented in `docs/deployment-notes.en.md` §6.

---

## [1.0.0] — 2026-09-01

First release. A deployment record for DeepSeek-V4-Flash-0731 on two NVIDIA GB10 (DGX Spark / GX10) nodes: configuration, self-checks, measurement tooling, and a set of conclusions that measurement later overturned.

### Included

- **Deployment notes** (`docs/deployment-notes.en.md`) — the full deployment path, current parameter values, capacity baselines, troubleshooting, and 26 methodology items.
- **Production configuration** — gmu 0.7935 / cudagraph accounting 1 / fused-markov 1 / DRAFT_CAPTURE_SIZES auto.
- **Patch A** (`patches/dspark_proposer.py`) — drafter-private cudagraph sizes, worth **+6%** on the single-stream baseline (count 81.0 → 85.7, struct 70.7 → 75.0, spread of about 1% across three runs). It is the only solid performance gain in the whole tuning round. **`VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto` must be set alongside it**, or installing the patch does nothing.
- **Launch script** (`scripts/dsv4f-launch.sh`) — preflight → launch → six self-checks → optional benchmark. The checks include proposer md5 consistency across both nodes and Patch A's switch state.
- **Measurement tooling** — single-stream baseline, concurrency sweep, TTFT/prefill, and a warm-keeping cron. Configuration comes from a profile file or the environment, so the tools can be pointed at other models.
- **Outstanding work** (`TODO.en.md`) — six unfinished measurements plus an evidence-strength ledger for the existing conclusions.

### Conclusions overturned during deployment

These are the notes' main value; the full record is in §8:

- **The `grep -q` + `pipefail` bug in the self-checks** — `grep -q` exits the moment it matches and closes the pipe, upstream `docker logs` is killed by SIGPIPE (exit 141), and `pipefail` then judges the whole pipeline failed. One check therefore **always passed silently** and another **always false-alarmed**, depending on whether `&&` or `||` was hung off it. Replaced with `grep -c`.
- **Both prefill scheduling parameters disproven** — `max_num_partial_prefills` has zero references on the V1 code path and is a V0 leftover; `long_prefill_token_threshold` behaves opposite to its name, being a hard per-step per-request token cap — a latency-for-throughput trade, not a free optimization.
- **Narrowing cudagraph capture sizes is entirely neutral** — both the benefit and the cost sit below the noise floor, confirmed by a revert experiment.
- **"Differing image IDs across nodes need fixing" is a non-issue** — each node runs its own `docker build`, so IDs are inherently irreproducible; the criterion is the md5 of files *inside* the image, not the image ID.
- **The KV pool is not deterministic** — it varies up to 11% between boots at identical configuration, which invalidated two conclusions previously drawn from it.
- **The client topology omitted a proxy layer in the middle** — that layer carries a switch that merges `reasoning` into `content`, and when it is on the symptom is indistinguishable from a server-side leak.
- **The `reasoning_effort` conclusion was backwards** — it *is* the thinking-mode switch (token spend 224 → 398). The original test was wrong because it compared only `content`, while the entire change lived in the `reasoning` field.

### Known limitations

- **§5.6's prefill and TTFT figures are upstream's, not measured here** — see [TODO item 1](TODO.en.md).
- **None of the four Python measurement tools has completed a run against a live service** — see [TODO item 0](TODO.en.md).

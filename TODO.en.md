[中文](TODO.md) · **English**

# Measurements still outstanding

Ordered by **whether the result could change a decision**, not by effort. Each item states why it matters, how to run it, where the result goes, and **what outcome would trigger what action**.

The first three could change an existing conclusion or the configuration; the last three strengthen evidence, and whether they are worth doing depends on how much the uncertainty bothers you.

---

## 0. Validate the tooling first ⬅ everything else depends on it

> ### ⚠️ None of the four Python tools has completed a run against a live service
>
> They share the request and parsing path in `_common.py`, and that code has only had syntax, config-loading, and concurrency-guard testing — **no script has ever completed a real request**; `measure-prefill.py` has never executed at all.
>
> **Reaching the network call is not the same as completing one.** This is precisely the class of mistake these notes keep warning about — a verification method that does not cover the kind of change made (notes [§10, item 25](docs/deployment-notes.en.md#10-methodology)).

**On the first run against a live service, confirm in this order:**

```bash
PROFILE=dsv4f ./scripts/bench-baseline.py bench --md    # simplest; validates the request path
PROFILE=dsv4f ./scripts/loadtest.py uniform 2           # then the concurrency path
PROFILE=dsv4f ./scripts/measure-prefill.py ttft --md    # finally the new script
```

**What specifically has not been verified:**

| Location | Unverified | Failure mode |
|---|---|---|
| `_common.chat()` | Whether the returned field names match the actual response | KeyError |
| `_common.chat_ttft()` | Streaming SSE parsing and first-chunk detection | TTFT always `None`, table full of "—" |
| `_common.engine_metrics()` | Whether it parses the `Avg prompt throughput` line correctly | Engine prefill column all "—" |
| `_common.long_prompt()` | Whether generated prompts land near the target token count | Length tiers misnamed; `4 chars ≈ 1 token` is only an estimate |
| `measure-prefill.py ttft` | Whether `max_tokens=64` reliably triggers a first chunk | Some rounds yield no TTFT |
| `--md` on both scripts | The actual markdown output format | Tables misalign when pasted into the notes |

**Fix problems on the spot and record the fixes in the CHANGELOG** — these are tool defects, not measurement results.

---

## 1. Prefill and TTFT

**Status**: the 8K→1,513 / 32K→2,284 / 100K→2,639 tok/s figures cited in [§5.6](docs/deployment-notes.en.md#56-prefill--not-benchmarked-on-this-machine) are **upstream's data, never measured here**. This is the only performance section still relying on external numbers.

**Why it matters**: TTFT is what users actually feel in a long-context agent workload. There is currently no local basis for "how long until the first token on a 60K prompt".

```bash
PROFILE=dsv4f ./scripts/measure-prefill.py ttft --md
```

Sweeps 1K / 8K / 32K / 100K, median of 3 per length, and also reads the engine's own `Avg prompt throughput`.

**Result goes into** §5.6, replacing the upstream figures, recorded with the configuration quadruple and the date.

**What outcome triggers what action**:

- If "faster with depth" holds here too → upstream's conclusion is confirmed and long contexts need no special handling
- If it does not (TTFT growing superlinearly with length) → long-context usage needs re-evaluation, possibly capping the practical range of `MAX_MODEL_LEN`

---

## 2. How long a short request waits behind a long prefill

**Status**: [§8.4](docs/deployment-notes.en.md#84-disproven-the-two-prefill-scheduling-parameters) derived `long_prefill_token_threshold`'s mechanism from source (a hard per-step token cap turning a 60K prefill from ~8 steps into ~59) and on that basis judged it "a latency-for-throughput trade".

**But the other half of that trade was never measured** — without the parameter, how long does a short request actually wait behind a long prefill? Absent that number, "worth it or not" is only inference.

```bash
PROFILE=dsv4f ./scripts/measure-prefill.py blocking --md
```

Measures short-request TTFT while idle as a baseline, then again with an 8K / 32K / 100K prefill in flight, and reports the ratio.

**Result goes into** §5.6, and then back to update §8.4's confidence level.

**What outcome triggers what action**:

- **Ratio near 1** → chunked prefill already lets short requests interleave, `--long-prefill-token-threshold` is **definitively unnecessary**, and §8.4's "disproven" is upgraded from mechanism inference to measured confirmation
- **Ratio large** (say 5× or more) → long prefill is monopolizing scheduling steps and the parameter **deserves reconsideration**, via its own restart plus a mixed-load A/B

---

## 3. `VLLM_USE_B12X_SPARSE_INDEXER=1`

**Status**: [§8.7](docs/deployment-notes.en.md#87-an-untested-candidate). This is the **only** B12X switch with neither a documented warning nor upstream campaign coverage. All gates are confirmed to pass: `is_device_capability_family(120)` returns True for SM121, and `use_fp4_indexer_cache` defaults to False.

**Why it is worth trying**: the same family's `VLLM_USE_B12X_MOE` is worth 55+ vs 29 tok/s — nearly 2×.

> ### ⚠️ Validation must compare output correctness, not just speed
>
> The B12X family has a precedent for **running fine while computing wrong results**: `VLLM_DSV4_B12X_COMPRESSED_MLA=1` produced wrong output plus a CUDA assert in upstream campaign E12.
>
> Method: same prompts, `temperature=0`, one run with the switch off and one on, **comparing `content` verbatim**. Faster but different output = revert immediately.

**Result goes into** §8.1's summary table plus §8.6/8.7.

---

## 4. A rigorous A/B for `FUSED_MARKOV_ARGMAX`

**Status**: currently recorded as "neutral", but the **evidence is asymmetric** — the `FUSED=0` arm has only 1 sample against 3 for `FUSED=1`. The notes already state honestly that it is kept for "theoretical bandwidth saving plus measured no harm", **not because it measured faster**.

**Why it may not be worth doing**: upstream reports +1.8% single-stream, while this machine's single-stream noise floor is ±1% — an effect only twice the floor, needing a dozen-plus rounds per arm to resolve.

```bash
# at least 5 rounds per arm, restarting between switch states
PROFILE=dsv4f ./scripts/bench-baseline.py warmup
PROFILE=dsv4f ./scripts/bench-baseline.py bench --md
```

**What outcome triggers what action**:

- Confirmed neutral → consider reverting to the default `0`, removing one unproven divergence
- Confirmed beneficial → replace "neutral" in §8.1 with the measured figure

---

## 5. The page-cache hypothesis for KV pool variance

**Status**: [§5.7](docs/deployment-notes.en.md#57-kv-pool-expected-values-and-noise) records **11% variance between boots** at identical configuration, but **the cause was never isolated**. Page-cache residue was suspected, but that restart also changed cudagraph accounting, and the two factors pushed the pool in opposite directions — no attribution possible.

**How to test**: a control set where **only cache state differs** — configuration held constant, one boot after `drop_caches` and one without, several repetitions each.

**Cost**: about 6 minutes per boot, at least 6 boots for a readable result → roughly 40 minutes of machine time.

**What outcome triggers what action**:

- If page cache is confirmed → the `drop_caches` step in the startup procedure gains a measured basis (currently it is copied from upstream's recipe)
- If not → the 11% has another cause worth chasing, because it determines how many samples every KV-related conclusion needs

---

## 6. The next gmu rung, 0.8010

**Status**: [§8.2](docs/deployment-notes.en.md#82-gmu-and-cudagraph-accounting-must-change-together) records that at the current 0.7935 the engine suggests 0.8010 as the next rung (accounting on), which converts to about 0.7935 in the accounting-off frame — still below the campaign's 0.80 physical edge.

**Why it was skipped**: diminishing returns, while KV pool measurement noise is 11% — the expected gain would most likely drown in it.

**If attempted**: multiple boots and a median are mandatory, or it cannot be resolved at all. That is exactly §5.7's lesson.

---

# Conclusions with known-weak evidence

**Not action items — a reminder** of how far each conclusion is actually supported:

| Conclusion | Strength | What's missing |
|---|---|---|
| Patch A +6% single-stream | **Solid** | — spread of 1% across three runs, well clear of noise |
| Patch A positive under concurrency | **Directional** | Concurrency metrics carry ±11% plus occasional 2× outliers; three rounds only support a direction |
| `FUSED_MARKOV=1` is neutral | **Weak** | Only 1 sample on the off arm (item 4 above) |
| The two prefill params are disproven | **Mechanism solid, cost unmeasured** | `max_num_partial_prefills` having zero V1 references is a hard fact; "how long short requests wait" is not measured (item 2) |
| Narrowing cudagraph sizes is neutral | **Solid** | — both benefit and cost below the noise floor, and verified by a revert experiment |
| Prefill gets faster with depth | **No local data** | The cited figures are upstream's (item 1) |

---

# Measurement discipline reminder

Before running any of the above, read [§6](docs/deployment-notes.en.md#6-measurement-discipline). The three easiest to forget:

1. **Take the median of at least three runs** — this machine produces 2× transient outliers
2. **Every reported number needs the configuration quadruple, acceptance length, and temperature**
3. **Know your chosen metric's noise floor first** — when the effect is smaller than the floor, no number of repetitions will produce anything but "no difference"

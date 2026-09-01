#!/usr/bin/env python3
"""预填充与 TTFT 测量。

为什么不用 max_tokens=1 隔离预填充：那是合成条件，不反映真实使用。
开着 chunked prefill 时预填充本来就和解码共享调度步，隔离测出来的数字
偏乐观。这里改用两个都在真实条件下成立的量：

  1. TTFT —— 用户直接感受到的量。必须用流式测（首个 chunk 到达时间）。
     这和"测吞吐必须非流式"不冲突：吞吐要数 token，而流式 chunk 装的是
     整步接受的多个 token；TTFT 只关心第一个 chunk 何时到达。

  2. vLLM 自己报的 Avg prompt throughput —— 引擎内部就把预填充和解码
     分开统计了，不需要任何合成设置。

用法：
  ./measure-prefill.py ttft              扫提示词长度，测 TTFT 与预填充吞吐
  ./measure-prefill.py blocking          长预填充在途时，短请求被堵多久
  ./measure-prefill.py ttft --md         同时输出 markdown 表格

配置见 _common.py。换模型只需改环境变量或 PROFILE。
"""
import statistics
import sys
import threading
import time

from _common import (CFG, banner, chat, chat_ttft, engine_metrics, long_prompt,
                     md_table)

# 扫描的提示词长度（目标 token 数）。要验证"预填充随深度变快"需要跨一个
# 数量级以上；短的那档同时充当对照。
LENGTHS = [1_000, 8_000, 32_000, 100_000]

ROUNDS = 3          # 任何性能结论至少 3 次取中位数（第 6.4 节）
SHORT_PROMPT = "What is 17 * 23? Answer with the number only."


def _fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def mode_ttft(emit_md):
    """扫提示词长度，测 TTFT。同时读引擎自己报的预填充吞吐。"""
    print(f"配置: {banner()}")
    print(f"每档 {ROUNDS} 次取中位数，max_tokens=64（只需触发首个 chunk）\n")
    print(f"{'目标长度':>10s} {'实际 prompt_tok':>16s} {'TTFT 中位':>10s} "
          f"{'TTFT 范围':>16s} {'引擎预填充峰值':>16s}")

    rows = []
    for target in LENGTHS:
        prompt = long_prompt(target)
        ttfts, inp = [], None
        for _ in range(ROUNDS):
            # 先用非流式拿一次 prompt_tokens（流式响应默认不带 usage）
            if inp is None:
                inp = chat(prompt, max_tokens=1)["inp"]
            r = chat_ttft(prompt, max_tokens=64)
            if r["ttft"] is not None:
                ttfts.append(r["ttft"])
            time.sleep(2)

        if not ttfts:
            print(f"{target:>10,d} {'失败':>16s}")
            continue

        em = engine_metrics(since_seconds=120)
        peak = em["prompt_peak"] if em else None
        med = statistics.median(ttfts)
        rng = f"{min(ttfts):.2f}–{max(ttfts):.2f}s"
        print(f"{target:>10,d} {inp:>16,d} {med:>9.2f}s {rng:>16s} "
              f"{_fmt(peak, 0) + ' tok/s' if peak else '—':>16s}")
        rows.append([f"{inp:,}", f"{med:.2f}s", rng,
                     f"{peak:.0f}" if peak else "—"])

    if emit_md and rows:
        print("\n--- markdown ---\n")
        print(md_table(["prompt_tokens", "TTFT 中位", "TTFT 范围",
                        "引擎预填充峰值 tok/s"], rows))


def mode_blocking(emit_md):
    """长预填充在途时，短请求要等多久。

    这是混合负载的真问题，也是判断预填充调度参数值不值的唯一依据——
    笔记第 8.4 节从源码推出了 long_prefill_token_threshold 的机制，
    但"短请求被堵多久"这一侧从未量过。
    """
    print(f"配置: {banner()}")
    print(f"每组 {ROUNDS} 次取中位数\n")

    # 基准：空闲时短请求的 TTFT
    base = []
    for _ in range(ROUNDS):
        base.append(chat_ttft(SHORT_PROMPT, max_tokens=16)["ttft"])
        time.sleep(1)
    base = [b for b in base if b is not None]
    base_med = statistics.median(base) if base else None
    print(f"  空闲时短请求 TTFT 中位: {_fmt(base_med, 3)}s")

    rows = [["空闲（基准）", "—", f"{base_med:.3f}s" if base_med else "—", "—"]]

    for target in (8_000, 32_000, 100_000):
        prompt = long_prompt(target)
        blocked = []
        inp_seen = []

        for _ in range(ROUNDS):
            done = {}

            def long_req():
                done["r"] = chat(prompt, max_tokens=32)

            t = threading.Thread(target=long_req)
            t.start()
            time.sleep(0.4)        # 让长请求先进入预填充
            r = chat_ttft(SHORT_PROMPT, max_tokens=16)
            if r["ttft"] is not None:
                blocked.append(r["ttft"])
            t.join()
            if "r" in done:
                inp_seen.append(done["r"]["inp"])
            time.sleep(2)

        if not blocked:
            continue
        med = statistics.median(blocked)
        ratio = f"{med / base_med:.1f}×" if base_med else "—"
        inp = int(statistics.median(inp_seen)) if inp_seen else 0
        print(f"  长请求 {inp:>7,d} tok 在途 → 短请求 TTFT 中位 "
              f"{med:6.3f}s  ({ratio} 于空闲)")
        rows.append([f"长请求 {inp:,} tok 在途", f"{inp:,}",
                     f"{med:.3f}s", ratio])

    print("\n  判读：比值接近 1 说明 chunked prefill 有效让短请求插进去了；"
          "\n        比值很大说明长预填充在独占调度步，此时"
          "\n        --long-prefill-token-threshold 才可能值得考虑（第 8.4 节）。")

    if emit_md:
        print("\n--- markdown ---\n")
        print(md_table(["场景", "长请求 prompt_tokens", "短请求 TTFT 中位",
                        "相对空闲"], rows))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    emit_md = "--md" in sys.argv
    mode = args[0] if args else "ttft"

    if mode == "ttft":
        mode_ttft(emit_md)
    elif mode == "blocking":
        mode_blocking(emit_md)
    else:
        print(__doc__)
        sys.exit(1)

#!/usr/bin/env python3
"""单流基线。

强制 stream=false 并读 usage.completion_tokens —— 投机解码下每个 decode step
只发一个 SSE chunk，数流式 delta 测的是 steps/s 而非 tokens/s，低报约 4 倍。
详见 docs/deployment-notes.md 第 6 节（测量纪律）。

用法：
  ./bench-baseline.py warmup       5 轮 × ~500 token，预热到稳态
  ./bench-baseline.py bench        四项标准测试
  ./bench-baseline.py bench --md   同时输出 markdown 表格

配置见 _common.py：PROFILE / ENDPOINT / MODEL 等。
参考基线来自 profile；没有基线时不打印偏差列，换个模型也能直接用。
"""
import sys

from _common import CFG, banner, chat, md_table

# 规范提示词 —— 与 dsv4f-launch.sh 的 bench() 逐字相同，也是产生笔记
# 第 5 节基线的那一组。改动任何一条都会让历史数据失去可比性。
TESTS = {
    "count":  ("Count from 1 to 300, separated by commas.", 600),
    "struct": ("Output a JSON array of 40 objects, each with fields "
               "id, name, email, active.", 600),
    "code":   ("Write a Python function that implements a red-black tree "
               "with insert and delete.", 600),
    "prose":  ("Write a 400-word essay on why distributed systems are hard.", 600),
}

# 为空则不打印偏差列 —— 换个模型时不必先有基线。
REFERENCE = CFG.get("reference") or {}

WARMUP_ROUNDS = 5
WARMUP_TOKENS = 900          # 笔记第 6.2 节：几次 100 token 的短预热不够，
                             # 需要 500–700 token 级别的生成才到稳态


def warmup():
    print(f"配置: {banner()}")
    for i in range(WARMUP_ROUNDS):
        r = chat(f"[warmup {i}] Count from 1 to 250, separated by spaces.",
                 WARMUP_TOKENS)
        print(f"  warmup {i + 1}/{WARMUP_ROUNDS}: {r['out']:5d} tok  "
              f"{r['wall']:6.2f}s  {r['tps']:6.1f} tok/s", flush=True)


def bench(emit_md):
    print(f"配置: {banner()}")
    has_ref = bool(REFERENCE)

    if has_ref:
        print(f"{'测试':8s} {'tokens':>7s} {'秒':>7s} {'tok/s':>8s} "
              f"{'参考':>7s} {'偏差':>8s}")
    else:
        print(f"{'测试':8s} {'tokens':>7s} {'秒':>7s} {'tok/s':>8s}")

    rows = []
    for name, (prompt, mt) in TESTS.items():
        r = chat(prompt, mt)
        ref = REFERENCE.get(name)
        if ref:
            dev = (r["tps"] - ref) / ref * 100
            print(f"{name:8s} {r['out']:7d} {r['wall']:7.2f} {r['tps']:8.1f} "
                  f"{ref:7.1f} {dev:+7.1f}%", flush=True)
            rows.append([name, r["out"], f"{r['wall']:.2f}",
                         f"{r['tps']:.1f}", f"{ref:.1f}", f"{dev:+.1f}%"])
        else:
            print(f"{name:8s} {r['out']:7d} {r['wall']:7.2f} {r['tps']:8.1f}",
                  flush=True)
            rows.append([name, r["out"], f"{r['wall']:.2f}", f"{r['tps']:.1f}"])

    if emit_md:
        headers = (["测试", "tokens", "秒", "tok/s", "参考", "偏差"] if has_ref
                   else ["测试", "tokens", "秒", "tok/s"])
        print("\n--- markdown ---\n")
        print(md_table(headers, rows))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    emit_md = "--md" in sys.argv
    mode = args[0] if args else "bench"

    if mode == "warmup":
        warmup()
    elif mode == "bench":
        bench(emit_md)
    else:
        print(__doc__)
        sys.exit(1)

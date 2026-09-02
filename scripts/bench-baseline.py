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
import json
import os
import sys

from _common import CFG, banner, chat, chat_text, md_table

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


# 正确性比对用的固定提示词。刻意选可判对错、且输出短而确定的题目。
VERIFY_PROMPTS = {
    "arith":  "What is 127 multiplied by 43? Reply with only the number.",
    "primes": "List the first 5 prime numbers, comma separated, no other text.",
    "logic":  "A farmer has 17 sheep. All but 9 run away. How many are left? "
              "Explain in one sentence.",
    "recall": "What is the chemical symbol for gold? Reply with only the symbol.",
}
# 选题标准：**短、答案唯一、指令无歧义、且不存在近似平局**。四条都要满足。
#
# 2026-09-02 标定：这四条各连跑 8 次，每条都只产生 1 种输出 —— 本部署在
# temperature=0 下是可复现的，逐字比对这个方法成立。但选题极易踩坑：
#
# 被淘汰的两条，说明这个标准不是形式主义：
#
# - "写一个合并有序列表的函数"：同一配置连跑两次，第 788 字符处稳定出现
#   docstring 空行差异。长自由生成本身就不确定，用它比对会产生假阳性
#   —— 这正是 2026-09-02 评估 B12X 稀疏索引器时误报"输出不一致"的原因。
# - "续写 2, 4, 8, 16 五个数"：模型有时回显原序列、有时只给续写部分。
#   指令有歧义，不是数值问题。
# - "列 5 个大于 100 的素数"：三次给出三组都正确但不同的答案
#   （101.. / 103.. / 1009..）。**答案正确不等于答案唯一** —— 这条最隐蔽，
#   看着像个精确的题目，实际有无穷多个正确解。
# - "255 的十六进制"：8 次里 'ff' 和 'FF' 各 4 次。答案唯一、指令也无歧义，
#   但**大小写构成了一个近似平局**，贪心解码在这种地方会翻转。
VERIFY_FILE = os.path.expanduser("~/.dsv4f-verify-%s.json")


def verify(save):
    """比对输出与基准，用于换内核/换开关后确认没有算错。

    首次用 --save 存基准；之后每次改配置跑一遍，逐字比对。
    温度 0、不加 nonce，所以输出应当逐字可复现。
    """
    banner()
    path = VERIFY_FILE % CFG.get("model", "model")
    ref = {}
    if not save:
        try:
            ref = json.load(open(path, encoding="utf-8"))
        except OSError:
            print(f"没有基准文件 {path}")
            print("先跑一次 verify --save 建立基准。")
            sys.exit(1)

    cur, bad = {}, 0
    for name, prompt in VERIFY_PROMPTS.items():
        cur[name] = chat_text(prompt, 256)
        if save:
            print(f"  {name:7s} {cur[name][:70]!r}")
            continue
        same = cur[name] == ref.get(name)
        print(f"  {name:7s} {'一致' if same else '不一致'}")
        if not same:
            bad += 1
            a, b = ref.get(name, ""), cur[name]
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                     min(len(a), len(b)))
            print(f"    首个差异在第 {i} 字符（基准 {len(a)} 字 / 现在 {len(b)} 字）")
            print(f"    基准 ...{a[max(0, i-50):i+50]!r}")
            print(f"    现在 ...{b[max(0, i-50):i+50]!r}")

    if save:
        json.dump(cur, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        print()
        print(f"基准已保存 -> {path}")
    else:
        print()
        print("全部一致" if bad == 0 else f"{bad} 项不一致")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    emit_md = "--md" in sys.argv
    mode = args[0] if args else "bench"

    if mode == "warmup":
        warmup()
    elif mode == "bench":
        bench(emit_md)
    elif mode == "verify":
        verify("--save" in sys.argv)
    else:
        print(__doc__)
        sys.exit(1)

#!/usr/bin/env python3
"""DSV4F 并发压测。

纪律（详见 docs/deployment-notes.md 第 6 节）：
  - 强制 stream=false。投机解码下每个 decode step 只发一个 SSE chunk，
    数流式 delta 测的是 steps/s 而非 tokens/s，低报约 4 倍。
  - 只读 usage.completion_tokens。
  - t=0，除非显式指定；T>0 的接受率惩罚是真实的，报数字必须说明。

用法:
  ./loadtest.py uniform 4          # 并发 4，同构提示词（对标上游 c4）
  ./loadtest.py mixed 6            # 并发 6，长预填充 + 短交互混合
  ./loadtest.py sweep              # 1/2/4/6 全扫

环境变量：ENDPOINT（默认 http://127.0.0.1:8078）、MODEL（默认 deepseek-v4-flash）
"""
import json, os, statistics, sys, threading, time, urllib.request

from _common import CFG

URL = CFG["url"]
MODEL = CFG["model"]

# 长提示词：约 6000 token 的真实预填充负载，用确定性内容保证可重复
_PARA = ("The scheduler assigns each request a token budget per step, and "
         "chunked prefill splits long prompts across multiple steps so that "
         "decode requests are not starved. ")
LONG_PROMPT = (_PARA * 320) + "\n\nSummarize the paragraph above in one sentence."
SHORT_PROMPT = "What is 17 * 23? Answer with the number only."
MED_PROMPT = "List 12 common causes of high tail latency in inference servers."


def _nonce() -> str:
    """前缀缓存按前缀匹配，nonce 必须在提示词最前面。

    不加 nonce 时，第二次跑同一提示词会命中前缀缓存，墙钟从 5.58s 掉到
    0.99s——测的就不再是预填充路径了（与 keepalive 同理，笔记第 6 节）。
    """
    return f"[run {os.urandom(6).hex()}] "


def call(prompt, max_tokens, temp=0.0):
    prompt = _nonce() + prompt
    body = json.dumps({
        "model": MODEL, "stream": False, "max_tokens": max_tokens,
        "temperature": temp,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    except Exception as e:
        return {"ok": False, "err": str(e)[:80], "dt": time.time() - t0}
    dt = time.time() - t0
    u = d["usage"]
    return {"ok": True, "dt": dt,
            "out": u["completion_tokens"], "inp": u["prompt_tokens"],
            "tps": u["completion_tokens"] / dt if dt else 0.0}


# MAX_NUM_SEQS=6 是引擎硬上限（日志实测 Running 从不超过 6）。
# 并发 > 6 只是在测「突发投放的尾巴效应」：前 6 个按 c6 速度跑完，剩下的
# 在低并发下单独跑，把墙钟拉长、聚合吞吐被稀释。那个数字看起来像
# 「并发越高越慢」，但它不是引擎容量，是测试形态的产物 —— 会招致误读。
# 档位只取 1/2/4/6。
MAX_NUM_SEQS = CFG["max_num_seqs"]


def _check(c):
    if c > MAX_NUM_SEQS:
        print(f"拒绝：c={c} 超过 MAX_NUM_SEQS={MAX_NUM_SEQS}。")
        print("超限并发测的是突发尾巴效应，不是引擎容量（笔记第 5 节）。")
        print("如果确实要测排队行为，请改这个守卫并说明意图。")
        raise SystemExit(1)


def run(jobs, label):
    """jobs: [(prompt, max_tokens)]。所有线程同时起跑。"""
    n = len(jobs)
    results = [None] * n
    barrier = threading.Barrier(n)

    def worker(i):
        prompt, mt = jobs[i]
        barrier.wait()
        results[i] = call(prompt, mt)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    ok = [r for r in results if r and r["ok"]]
    bad = [r for r in results if r and not r["ok"]]
    if not ok:
        print(f"{label}: 全部失败 {bad[:1]}")
        return

    total_out = sum(r["out"] for r in ok)
    total_inp = sum(r["inp"] for r in ok)
    per = sorted(r["tps"] for r in ok)
    lat = sorted(r["dt"] for r in ok)
    p95 = lat[min(len(lat) - 1, max(0, round(0.95 * len(lat)) - 1))]
    # 注意：这不是预填充吞吐。分母是总墙钟，包含解码时间，所以会严重
    # 低报真实的 prefill 速度。它只能用来横向比较不同并发下的相对变化。
    # 要测真正的预填充：发长提示词并设 max_tokens=1，或读 vLLM 引擎日志里的
    # "Avg prompt throughput"。详见 docs/deployment-notes.md 第 5.6 节。
    print(f"{label:16s} n={n} 墙钟={wall:6.2f}s  "
          f"解码聚合={total_out/wall:7.1f} tok/s  "
          f"单流中位={statistics.median(per):6.1f}  "
          f"输入吞吐={total_inp/wall:8.1f} tok/s (in={total_inp})  "
          f"p50={statistics.median(lat):5.2f}s p95={p95:5.2f}s"
          + (f"  失败={len(bad)}" if bad else ""))


def uniform(c):
    _check(c)
    run([(MED_PROMPT, 600)] * c, f"uniform c{c}")


def mixed(c):
    _check(c)
    """1 个长预填充 + 其余短/中交互 —— 对标真实混合负载。"""
    jobs = [(LONG_PROMPT, 200)]
    for i in range(c - 1):
        jobs.append((SHORT_PROMPT, 60) if i % 2 == 0 else (MED_PROMPT, 400))
    run(jobs, f"mixed c{c}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        for c in (1, 2, 4, 6):
            uniform(c)
            time.sleep(3)
        print()
        for c in (2, 4, 6):
            mixed(c)
            time.sleep(3)
    elif mode == "uniform":
        uniform(int(sys.argv[2]))
    elif mode == "mixed":
        mixed(int(sys.argv[2]))

#!/usr/bin/env python3
"""测量脚本的共享配置与工具。

设计目标是**换个模型也能用**：所有与具体部署绑定的值都从环境变量或
profile 文件读，脚本本身不含硬编码。

配置来源（后者覆盖前者）：
  1. 内置默认值
  2. profile 文件            scripts/profiles/<name>.json，用 PROFILE=<name> 选择
  3. 环境变量                ENDPOINT / MODEL / MAX_NUM_SEQS / SPECULATIVE

用法示例：
  PROFILE=dsv4f ./bench-baseline.py bench
  ENDPOINT=http://host:8000 MODEL=qwen3-32b MAX_NUM_SEQS=16 ./loadtest.py sweep
"""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

DEFAULTS = {
    "endpoint": "http://127.0.0.1:8078",
    "model": "deepseek-v4-flash",
    # 并发守卫上限。超过引擎的 max_num_seqs 时，测到的是突发投放的尾巴效应，
    # 不是引擎容量（见 docs/deployment-notes.md 第 5.3 节）。
    "max_num_seqs": 6,
    # 模型是否使用投机解码。False 时跳过接受长度相关的采集与展示。
    "speculative": True,
    # 容器名，仅用于从引擎日志读指标；留空则跳过日志采集。
    "container": "",
    # 四项标准测试的参考基线，用于打印偏差。为空则不打印偏差列。
    "reference": {},
}


def load_config():
    cfg = dict(DEFAULTS)

    name = os.getenv("PROFILE", "").strip()
    if name:
        p = Path(__file__).parent / "profiles" / f"{name}.json"
        if not p.exists():
            raise SystemExit(f"找不到 profile: {p}")
        cfg.update(json.loads(p.read_text(encoding="utf-8")))

    for key, env in [("endpoint", "ENDPOINT"), ("model", "MODEL"),
                     ("container", "CONTAINER")]:
        v = os.getenv(env)
        if v:
            cfg[key] = v
    if os.getenv("MAX_NUM_SEQS"):
        cfg["max_num_seqs"] = int(os.environ["MAX_NUM_SEQS"])
    if os.getenv("SPECULATIVE"):
        cfg["speculative"] = os.environ["SPECULATIVE"].lower() in ("1", "true", "yes", "on")

    cfg["url"] = cfg["endpoint"].rstrip("/") + "/v1/chat/completions"
    return cfg


CFG = load_config()


def nonce() -> str:
    """前缀缓存按前缀匹配，所以 nonce 必须放在提示词最前面。

    不加时，同一长提示词第二次跑会命中缓存，墙钟可能从 5.58s 掉到 0.99s——
    测的就不再是预填充路径了（docs/deployment-notes.md 第 6.3 节）。
    """
    return f"[run {os.urandom(6).hex()}] "


def chat(prompt, max_tokens, temperature=0.0, prefix_nonce=True, timeout=900):
    """非流式请求。测吞吐必须用这个：投机解码下每个 decode step 只发一个
    SSE chunk，数流式 delta 测的是 steps/s 而非 tokens/s，低报约 4 倍。
    """
    body = json.dumps({
        "model": CFG["model"], "stream": False,
        "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "user",
                      "content": (nonce() if prefix_nonce else "") + prompt}],
    }).encode()
    req = urllib.request.Request(CFG["url"], data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    u = d["usage"]
    return {"wall": dt, "out": u["completion_tokens"], "inp": u["prompt_tokens"],
            "tps": u["completion_tokens"] / dt if dt else 0.0}


def chat_text(prompt, max_tokens=256, temperature=0.0, timeout=900):
    """返回模型输出的文本本身，用于正确性比对。

    与 chat() 的区别：chat() 只返回计时指标，做不了输出比对。默认不加
    nonce —— 比对要求逐字可复现，而 nonce 会改变提示词。
    """
    body = json.dumps({
        "model": CFG["model"], "stream": False,
        "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(CFG["url"], data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def chat_ttft(prompt, max_tokens, temperature=0.0, prefix_nonce=True, timeout=900):
    """流式请求，测 TTFT（首个 chunk 到达时间）。

    这里用流式是**正确的**，和"测吞吐必须非流式"不冲突——两者是不同指标：
    吞吐要数 token，流式 chunk 装的是整步接受的多个 token；
    TTFT 只关心第一个 chunk 什么时候到，流式才拿得到这个时刻。
    """
    body = json.dumps({
        "model": CFG["model"], "stream": True,
        "max_tokens": max_tokens, "temperature": temperature,
        "messages": [{"role": "user",
                      "content": (nonce() if prefix_nonce else "") + prompt}],
    }).encode()
    req = urllib.request.Request(CFG["url"], data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            delta = d["choices"][0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning"):
                chunks += 1
                if ttft is None:
                    ttft = time.time() - t0
    return {"ttft": ttft, "wall": time.time() - t0, "chunks": chunks}


def engine_metrics(since_seconds=60):
    """从引擎日志读 vLLM 自己算的 prompt / generation 吞吐。

    这是**引擎内部的分离统计**，不需要构造 max_tokens=1 之类的合成条件，
    在真实混合负载下照样报：

        Engine 000: Avg prompt throughput: 2.1 tokens/s,
                    Avg generation throughput: 49.5 tokens/s, ...

    返回窗口内的最大值——预填充是突发的，均值会被空闲区间稀释。
    """
    c = CFG.get("container")
    if not c:
        return None
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", f"{since_seconds}s", c],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=30).stdout
    except Exception:
        return None
    prompt_tp, gen_tp = [], []
    for line in out.splitlines():
        if "Avg prompt throughput:" not in line:
            continue
        try:
            prompt_tp.append(float(line.split("Avg prompt throughput:")[1]
                                   .split("tokens/s")[0].strip()))
            gen_tp.append(float(line.split("Avg generation throughput:")[1]
                                .split("tokens/s")[0].strip()))
        except (IndexError, ValueError):
            continue
    if not prompt_tp:
        return None
    return {"prompt_peak": max(prompt_tp), "gen_peak": max(gen_tp),
            "samples": len(prompt_tp)}


def acceptance(since_seconds=60):
    """从引擎日志读最近一次 mean acceptance length。非投机模型返回 None。"""
    if not CFG.get("speculative") or not CFG.get("container"):
        return None
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", f"{since_seconds}s", CFG["container"]],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=30).stdout
    except Exception:
        return None
    vals = []
    for line in out.splitlines():
        if "Mean acceptance length:" in line:
            try:
                vals.append(float(line.split("Mean acceptance length:")[1]
                                  .split(",")[0].strip()))
            except (IndexError, ValueError):
                continue
    return vals[-1] if vals else None


def long_prompt(target_tokens):
    """生成约 target_tokens 个 token 的确定性长提示词。

    内容固定，保证跨次可重复；实际 token 数由响应里的 prompt_tokens 为准。
    2026-09-02 标定：这段英文实测 **5.26 字符 ≈ 1 token**。
    系数 4 只得目标的 76%，系数 3 只得 57%（两次实测反推一致）。
    实际 token 数仍以响应里的 prompt_tokens 为准。
    """
    para = ("The scheduler assigns each request a token budget per step, and "
            "chunked prefill splits long prompts across multiple steps so that "
            "decode requests are not starved. ")
    reps = max(1, int(target_tokens * 5.26 / len(para)))
    return para * reps + "\n\nSummarize the paragraph above in one sentence."


def md_table(headers, rows):
    """输出 markdown 表格，可直接粘进笔记。"""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def banner():
    """打印本次测量的配置——报数字必须同时说明配置（第 6.5 节）。"""
    spec = "是" if CFG["speculative"] else "否"
    return (f"endpoint={CFG['endpoint']}  model={CFG['model']}  "
            f"max_num_seqs={CFG['max_num_seqs']}  投机解码={spec}")

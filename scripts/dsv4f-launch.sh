#!/usr/bin/env bash
#
# DeepSeek-V4-Flash-0731 on dual GX10 — 完整启动流程
#
#   前置检查 → 启动 → 六项自检 → 可选基线
#
# 用法:
#   ./dsv4f-launch.sh              完整流程
#   ./dsv4f-launch.sh --skip-fabric  跳过带宽测试（省 40 秒）
#   ./dsv4f-launch.sh --bench       启动后跑基线
#   ./dsv4f-launch.sh --check-only  只跑自检（服务已在运行时用）
#
set -uo pipefail

# ===================== 配置 =====================
REPO_DIR="${REPO_DIR:-$HOME/services/dsv4f}"
WORKER="${WORKER:-192.168.100.2}"
CONTAINER="${CONTAINER:-dsv4f-vllm-dspark-1}"
PORT="${PORT:-8078}"
MODEL="${MODEL:-deepseek-v4-flash}"

HCA_A="rocep1s0f1";   IP_A="192.168.100.2"
HCA_B="roceP2p1s0f1"; IP_B="192.168.101.2"
FABRIC_MIN_GBPS=184          # NVIDIA 官方验收门槛

SCHED_PY="/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
DSPARK_PY="/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py"
PROP_PY="/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark_proposer.py"

# ===================== 输出 =====================
R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; B=$'\e[1m'; N=$'\e[0m'
ok()   { echo "  ${G}✓${N} $*"; }
bad()  { echo "  ${R}✗${N} $*"; FAILED=1; }
warn() { echo "  ${Y}!${N} $*"; }
step() { echo; echo "${B}=== $* ===${N}"; }
FAILED=0

SKIP_FABRIC=0; RUN_BENCH=0; CHECK_ONLY=0
for a in "$@"; do case "$a" in
  --skip-fabric) SKIP_FABRIC=1 ;;
  --bench)       RUN_BENCH=1 ;;
  --check-only)  CHECK_ONLY=1; SKIP_FABRIC=1 ;;
  -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
esac; done

# ===================== 1. 前置检查 =====================
preflight() {
  step "1. 前置检查"

  ssh -o BatchMode=yes -o ConnectTimeout=5 "$WORKER" true 2>/dev/null \
    && ok "worker $WORKER 免密 ssh 可达" \
    || { bad "worker 不可达 — 检查 ssh 密钥和 RoCE 网络"; return 1; }

  # RoCE 地址与 MTU（netplan 应已持久化）
  for host in local "$WORKER"; do
    local tag out
    [ "$host" = local ] && { tag=head; out=$(ip -br addr); } \
                        || { tag=worker; out=$(ssh "$host" "ip -br addr"); }
    echo "$out" | grep -q "enp1s0f1np1.*192.168.100" \
      && ok "$tag rail A 地址就位" || bad "$tag rail A 无 IP — netplan 未生效？"
    echo "$out" | grep -q "enP2p1s0f1np1.*192.168.101" \
      && ok "$tag rail B 地址就位" || bad "$tag rail B 无 IP"
  done

  # GID 3 必须是 RoCE v2；idx 2 的 GID 值相同但走 v1，抄错会静默降级
  local t
  t=$(cat "/sys/class/infiniband/$HCA_A/ports/1/gid_attrs/types/3" 2>/dev/null)
  [ "$t" = "RoCE v2" ] && ok "GID index 3 = RoCE v2" || bad "GID 3 是 '$t'，应为 RoCE v2"

  # 残留容器 —— 上游明确要求全新 docker run，绝不 restart 旧容器
  for host in local "$WORKER"; do
    local tag n
    [ "$host" = local ] && { tag=head;   n=$(docker ps -aq -f "name=$CONTAINER" | wc -l); } \
                        || { tag=worker; n=$(ssh "$host" "docker ps -aq -f name=$CONTAINER | wc -l"); }
    [ "$n" -eq 0 ] && ok "$tag 无残留容器" \
                   || warn "$tag 有 $n 个残留容器，将被 compose 重建"
  done
}

# ===================== 2. Fabric 带宽 =====================
# 单接口最多 98 Gb/s —— 必须两个 PCIe 域同时压才能看到 196。
# 动过 QSFP 线后不重启会静默掉到 13 Gb/s（GB10 已知初始化 bug）。
fabric_check() {
  step "2. Fabric 带宽（双 rail 同时）"

  local a b tmp; tmp=$(mktemp -d)
  ssh "$WORKER" "pkill -f ib_write_bw" 2>/dev/null; sleep 1
  ssh -n "$WORKER" "nohup ib_write_bw -d $HCA_A -x 3 -F --report_gbits -D 15 -p 18515 >/dev/null 2>&1 &"
  ssh -n "$WORKER" "nohup ib_write_bw -d $HCA_B -x 3 -F --report_gbits -D 15 -p 18516 >/dev/null 2>&1 &"
  sleep 3
  ib_write_bw -d "$HCA_A" -x 3 -F --report_gbits -D 15 -p 18515 "$IP_A" 2>/dev/null \
    | awk '/^ [0-9]/{print $4}' > "$tmp/a" &
  ib_write_bw -d "$HCA_B" -x 3 -F --report_gbits -D 15 -p 18516 "$IP_B" 2>/dev/null \
    | awk '/^ [0-9]/{print $4}' > "$tmp/b" &
  wait
  a=$(cat "$tmp/a" 2>/dev/null); b=$(cat "$tmp/b" 2>/dev/null); rm -rf "$tmp"

  if [ -z "$a" ] || [ -z "$b" ]; then
    bad "带宽测试失败 — 确认两台都装了 perftest"
    return
  fi
  local total; total=$(awk -v x="$a" -v y="$b" 'BEGIN{printf "%.2f", x+y}')
  echo "     rail A = $a Gb/s   rail B = $b Gb/s"
  if awk -v t="$total" -v m="$FABRIC_MIN_GBPS" 'BEGIN{exit !(t>=m)}'; then
    ok "合计 $total Gb/s（≥ $FABRIC_MIN_GBPS）"
  else
    bad "合计仅 $total Gb/s"
    echo "     ${Y}若接近 26：动过 QSFP 线后未重启。接好线别再动，重启两台，重测。${N}"
  fi
}

# ===================== 3. 启动 =====================
launch() {
  step "3. 启动"
  echo "  清 page cache（上游复现配方要求）..."
  sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  ssh -t "$WORKER" "sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null

  echo "  启动中（首次约 8 分钟：155 GiB 权重 + JIT + cudagraph 捕获）..."
  cd "$REPO_DIR" || exit 1
  ./start-deepseek-v4-flash-dspark.sh || { bad "启动脚本失败"; return 1; }
  ok "服务已就绪"
}

# ===================== 4. 四项自检 =====================
# 本部署的核心风险不是崩溃，而是「能跑、质量正常、就是慢一半」。
selfcheck() {
  step "4. 自检"

  # 陷阱 1 — Patch 3：冷启动乱码的根因修复，位于调度器
  #          只在冷预填充发作，热请求 0/19 失败 → 冒烟测试测不出来
  local h w
  h=$(docker exec "$CONTAINER" grep -c is_prefill_chunk "$SCHED_PY" 2>/dev/null)
  w=$(ssh "$WORKER" "docker exec $CONTAINER grep -c is_prefill_chunk $SCHED_PY" 2>/dev/null)
  [ "$h" = 5 ] && ok "Patch 3 head ($h)"     || bad "Patch 3 head = '$h'，应为 5"
  [ "$w" = 5 ] && ok "Patch 3 worker ($w)"   || bad "Patch 3 worker = '$w'，应为 5"

  # 陷阱 2 — Patch 4：draft loader 静默丢 12 个 shared_experts 张量
  #          失效 → 接受率 60.2% 掉到 25.7%，32.7 而非 55.4 tok/s
  # 判据用文件内容，不用 docker logs：环形缓冲会覆盖启动期日志，而
  # 「无 w1_weight_scale_2 警告」方向相反 —— 日志被截断反而"通过"。
  # 失效是 0；实测在位为 3（笔记记的 2 是更早的文件版本）。
  h=$(docker exec "$CONTAINER" grep -c shared_experts.gate_up_proj "$DSPARK_PY" 2>/dev/null)
  w=$(ssh "$WORKER" "docker exec $CONTAINER grep -c shared_experts.gate_up_proj $DSPARK_PY" 2>/dev/null)
  [ "${h:-0}" -ge 2 ] 2>/dev/null && ok "Patch 4 head ($h)"   || bad "Patch 4 head = '$h'，应 >=2（0=失效）"
  [ "${w:-0}" -ge 2 ] 2>/dev/null && ok "Patch 4 worker ($w)" || bad "Patch 4 worker = '$w'，应 >=2"

  # 陷阱 3 — B12X MoE：=0 静默回落 DEEPGEMM_MXFP4，29 vs 55+ tok/s
  h=$(docker exec "$CONTAINER" printenv VLLM_USE_B12X_MOE 2>/dev/null)
  w=$(ssh "$WORKER" "docker exec $CONTAINER printenv VLLM_USE_B12X_MOE" 2>/dev/null)
  [ "$h" = 1 ] && ok "B12X MoE head"   || bad "B12X head = '$h'，应为 1"
  [ "$w" = 1 ] && ok "B12X MoE worker" || bad "B12X worker = '$w'，应为 1"
  n=$(docker logs "$CONTAINER" 2>&1 | grep -c "Using 'B12X' Mxfp4 MoE backend" || true)
  [ "${n:-0}" -gt 0 ] && ok "B12X 后端日志已确认" || warn "未见 B12X 后端日志（环形缓冲可能已覆盖，非故障判据）"

  # 陷阱 5 — Patch A 两台必须同一份代码，否则集合通信序列发散 → NCCL 挂起。
  #          判据是镜像内文件 md5，不是镜像 ID —— 两台各自 build，ID 天然不同。
  h=$(docker exec "$CONTAINER" md5sum "$PROP_PY" 2>/dev/null | cut -d" " -f1)
  w=$(ssh "$WORKER" "docker exec $CONTAINER md5sum $PROP_PY" 2>/dev/null | cut -d" " -f1)
  if [ -n "$h" ] && [ "$h" = "$w" ]; then
    ok "proposer 两台一致 (${h:0:8})"
  else
    bad "proposer 两台不一致！head=${h:0:8} worker=${w:0:8} — NCCL 会挂起"
  fi

  # Patch A 开关：off/空 = 补丁行为与原版逐字节相同，不算故障
  dcs=$(docker exec "$CONTAINER" printenv VLLM_DSPARK_DRAFT_CAPTURE_SIZES 2>/dev/null)
  case "${dcs:-off}" in
    ""|0|off|false|no) warn "Patch A 未启用 (DRAFT_CAPTURE_SIZES=${dcs:-未设})" ;;
    *) n=$(docker logs "$CONTAINER" 2>&1 | grep -c "drafter-private cudagraph capture sizes enabled" || true); [ "${n:-0}" -gt 0 ] && ok "Patch A 生效 (DRAFT_CAPTURE_SIZES=$dcs)" || bad "DRAFT_CAPTURE_SIZES=$dcs 但无生效日志 — 补丁可能没进镜像" ;;
  esac

  # 环境确认
  local kv pool ver
  ver=$(docker logs "$CONTAINER" 2>&1 | grep -oE "0\.21\.[0-9a-zA-Z.+]+g[0-9a-f]+" | head -1)
  kv=$(docker logs "$CONTAINER" 2>&1 | grep -oE "Using [a-z0-9_]+ data type to store kv cache" | head -1 | awk '{print $2}')
  pool=$(docker logs "$CONTAINER" 2>&1 | grep -oE "GPU KV cache size: [0-9,]+" | tail -1 | awk '{print $5}')
  [ -n "$ver" ]  && ok "vLLM $ver"
  [ -n "$kv" ]   && ok "KV dtype $kv"
  [ -n "$pool" ] && ok "KV 池 $pool token"

  # 内存/调度配置（2026-08-30 调优结果，笔记第 8 节）
  gmu=$(docker exec "$CONTAINER" cat /proc/1/cmdline | tr '\000' '\n' | grep -A1 -x -- '--gpu-memory-utilization' | tail -1)
  ecg=$(docker exec "$CONTAINER" printenv VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS 2>/dev/null)
  fma=$(docker exec "$CONTAINER" printenv VLLM_DSPARK_FUSED_MARKOV_ARGMAX 2>/dev/null)
  ok "gmu $gmu / cudagraph 记账 ${ecg:-0} / fused-markov ${fma:-0}"
  # gmu 的建议值取决于 cudagraph 记账开关，开着时不要抄关闭时打印的建议值
  if [ "$ecg" = 1 ]; then
    docker logs "$CONTAINER" 2>&1 | grep -oE 'increase [-][-]gpu-memory-utilization to [0-9.]+' | tail -1 | sed 's/^/  提示: 日志建议 /'
  fi

  # API 与工具调用
  curl -fsS --max-time 10 "http://127.0.0.1:$PORT/v1/models" >/dev/null \
    && ok "API 响应" || bad "API 无响应"

  curl -s --max-time 60 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"用工具查北京时间\"}],
         \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"查时间\",
         \"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}" \
    | grep -c '"tool_calls"' > /dev/null && ok "工具调用返回结构化 tool_calls" || bad "工具调用异常 — 检查 --tool-call-parser deepseek_v4"

  # 总闸 —— 前三项任何一个失效都会体现在这里
  echo "  预热并测接受长度..."
  curl -s --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"stream\":false,\"temperature\":0,\"max_tokens\":400,
         \"messages\":[{\"role\":\"user\",\"content\":\"[$RANDOM] Count from 1 to 200, separated by commas.\"}]}" \
    >/dev/null
  sleep 12
  local acc
  acc=$(docker logs --tail=100 "$CONTAINER" 2>&1 \
        | grep -oE "Mean acceptance length: [0-9.]+" | tail -1 | awk '{print $4}')
  if [ -z "$acc" ]; then
    warn "未取到接受长度（样本可能太少）"
  elif awk -v a="$acc" 'BEGIN{exit !(a>=3.5)}'; then
    ok "接受长度 $acc / 6（健康 ≥3.5）"
  else
    bad "接受长度仅 $acc — 存在静默失效，查上面三项"
  fi
}

# ===================== 5. 基线 =====================
# 必须非流式：投机解码下每 decode step 只发一个 SSE chunk，
# 数流式 delta 测的是 steps/s，同一请求低报 4 倍（14.7 vs 60.1）。
bench() {
  step "5. 基线（非流式，temperature 0，唯一 nonce）"
  local run
  run() {
    local name="$1" prompt="$2" t0 t1 out toks
    t0=$(date +%s.%N)
    out=$(curl -s --max-time 180 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"stream\":false,\"temperature\":0,\"max_tokens\":600,
           \"messages\":[{\"role\":\"user\",\"content\":\"[$RANDOM$RANDOM] $prompt\"}]}")
    t1=$(date +%s.%N)
    toks=$(echo "$out" | python3 -c "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)
    [ -z "$toks" ] && { echo "  $name: FAILED"; return; }
    printf "  %-8s %4s tok  %6.2fs  %s tok/s\n" "$name" "$toks" \
      "$(echo "$t1-$t0"|bc)" "$(echo "scale=1;$toks/($t1-$t0)"|bc)"
  }
  # 笔记第 6 节：热身衰减 30%，启动日志完全看不出来。几次 100 token 的短预热不够，
  # 需要 500–700 token 级别的生成才到稳态。
  echo "  预热（5 轮 × ~500 token，笔记第 6 节要求）..."
  for _ in 1 2 3 4 5; do
    run "warmup" "Count from 1 to 250, separated by spaces." >/dev/null
  done
  echo "  参考值: count 83.9 / struct 79.4 / code 67.8 / prose 34.0"
  run "count"  "Count from 1 to 300, separated by commas."
  run "struct" "Output a JSON array of 40 objects, each with fields id, name, email, active."
  run "code"   "Write a Python function that implements a red-black tree with insert and delete."
  run "prose"  "Write a 400-word essay on why distributed systems are hard."
}

# ===================== 主流程 =====================
echo "${B}DeepSeek-V4-Flash-0731 · dual GX10 · TP=2 · DSpark k=5${N}"

if [ "$CHECK_ONLY" = 0 ]; then
  preflight || exit 1
  [ "$SKIP_FABRIC" = 0 ] && fabric_check
  [ "$FAILED" = 1 ] && { echo; echo "${R}前置检查未通过，中止。${N}"; exit 1; }
  launch || exit 1
fi

selfcheck
[ "$RUN_BENCH" = 1 ] && bench

step "结果"
if [ "$FAILED" = 0 ]; then
  echo "  ${G}全部通过${N}   API: http://$(hostname -I | awk '{print $1}'):$PORT/v1"
else
  echo "  ${R}有检查未通过 — 见上方 ✗${N}"
  echo "  日志: docker logs --tail=200 $CONTAINER"
fi
exit "$FAILED"

#!/usr/bin/env bash
#
# DSV4F 保温 —— 防止闲置衰减
#
# 上游实测：40 分钟压测后闲置约 30 分钟，count300 从 83.5 掉到 60.4 tok/s；
# 大量预热后恢复。启动日志里看不出来，服务器全程报告正常、回答正确，就是慢。
# 短请求不管用 —— 需要 500-700 token 级别的生成才能回到稳态。
#
# 装法（在 head 节点，即 192.168.1.32 上）:
#   cp keepalive.sh ~/services/dsv4f/ && chmod +x ~/services/dsv4f/keepalive.sh
#   crontab -e
#   */15 * * * * /home/<user>/services/dsv4f/keepalive.sh
#
# 成本: 每 15 分钟约 7 秒 GPU 时间 = 0.8% 占空比。
#
set -uo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8078}"
MODEL="${MODEL:-deepseek-v4-flash}"
CONTAINER="${CONTAINER:-dsv4f-vllm-dspark-1}"
LOG="${LOG:-/var/tmp/dsv4f-keepalive.log}"
KUMA_PUSH_URL="${KUMA_PUSH_URL:-}"        # 可选，见文件末尾

# 服务没起来就安静退出（重启窗口、维护期间不刷日志）
curl -fsS --max-time 5 "$ENDPOINT/v1/models" >/dev/null 2>&1 || exit 0

# 正在跑真实请求就跳过 —— 不跟真实请求抢并发槽，而且有流量本来就是热的
RUNNING=$(curl -fsS --max-time 5 "$ENDPOINT/metrics" 2>/dev/null \
  | awk '/^vllm:num_requests_running/ {print int($2); exit}')
if [ "${RUNNING:-0}" -gt 0 ]; then
  echo "$(date -Is)  skip (running=$RUNNING)" >> "$LOG"
  exit 0
fi

# nonce 避开前缀缓存 —— 让预填充路径也一起保温，顺便让数字可比
NONCE="$RANDOM$RANDOM"
T0=$(date +%s.%N)
RESP=$(curl -s --max-time 120 "$ENDPOINT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":false,\"temperature\":0,\"max_tokens\":600,
       \"messages\":[{\"role\":\"user\",\"content\":\"[$NONCE] Count from 1 to 300, separated by commas.\"}]}")
T1=$(date +%s.%N)

TOKS=$(echo "$RESP" | python3 -c \
  "import sys,json;print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null)

if [ -z "$TOKS" ]; then
  echo "$(date -Is)  FAILED  $(echo "$RESP" | head -c 200)" >> "$LOG"
  exit 1
fi

SEC=$(echo "$T1-$T0" | bc)
RATE=$(echo "scale=1; $TOKS/($T1-$T0)" | bc)
ACC=$(docker logs --tail=60 "$CONTAINER" 2>&1 \
  | grep -oE "Mean acceptance length: [0-9.]+" | tail -1 | awk '{print $4}')

printf '%s  %s tok  %.2fs  %s tok/s  accept=%s\n' \
  "$(date -Is)" "$TOKS" "$SEC" "$RATE" "${ACC:-?}" >> "$LOG"

# 日志保留最近 2000 行（约三周）
if [ "$(wc -l < "$LOG")" -gt 2500 ]; then
  tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi

# 可选：推给 Uptime Kuma，把保温顺便当成性能监控
# 在 Kuma 里建一个 Push 监控，把生成的 URL 填进 KUMA_PUSH_URL，
# 心跳间隔设 900 秒以上（比 cron 间隔宽松），就能在图上看到 tok/s 长期曲线。
if [ -n "$KUMA_PUSH_URL" ]; then
  ST=up
  awk -v r="$RATE"     'BEGIN{exit !(r<50)}'  && ST=down
  awk -v a="${ACC:-9}" 'BEGIN{exit !(a<3.0)}' && ST=down
  curl -fsS --max-time 10 \
    "$KUMA_PUSH_URL?status=$ST&msg=${RATE}tok/s+acc${ACC:-?}&ping=${RATE}" \
    >/dev/null 2>&1
fi

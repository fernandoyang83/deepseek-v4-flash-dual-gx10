**中文** · [English](deployment-notes.en.md)

# DeepSeek-V4-Flash-0731 在两台 GX10 上的部署

这份文档记录一套已经跑起来的部署：**两台 NVIDIA GB10（DGX Spark / GX10），TP=2，DSpark 投机解码 k=5，NVFP4 KV cache**。

内容包括怎么装、怎么验证装对了、该期望什么性能、出问题时怎么定位，以及试过哪些调优、结论是什么。所有数字都是本机实测，测量方法在文中写明。

> ## ⚠️ 这份文档测的是设备的极限，不是模型的质量
>
> **全文没有任何能力评测** —— 没有 benchmark 分数、没有输出质量对比、没有"这个模型好不好用"的结论。所有内容只回答：**这套硬件加这套配置能撑住多少、在哪里到顶、什么情况下会静默地慢一半。**
>
> 这解释了第 5.1 节那四条测试提示词的选法 —— 它们按**接受率档位**挑，不按能力维度挑，作用是覆盖从"高度可预测"到"自由文本"的范围，好让吞吐数字可解释。**拿它们评价模型能力没有意义。**

**读之前还要知道两件事：**

- 这套配置优化的是**单流延迟**（agent 场景），不是高并发吞吐。拿它和四节点无投机的吞吐配置比绝对数字没有意义。
- 这个部署最大的风险不是崩溃，而是**"能跑、输出质量正常、就是慢一半"**。文档里相当篇幅在讲怎么发现这类静默失败。

---

## 目录

1. [这套部署是什么](#1-这套部署是什么)
2. [先读这个：吞吐模型](#2-先读这个吞吐模型)
3. [部署步骤](#3-部署步骤)
4. [参数参考](#4-参数参考)
5. [性能基线与容量规划](#5-性能基线与容量规划)
6. [测量纪律](#6-测量纪律)
7. [故障模式](#7-故障模式)
8. [调优记录：试过什么，结论是什么](#8-调优记录试过什么结论是什么)
9. [客户端集成注意事项](#9-客户端集成注意事项)
10. [方法论](#10-方法论)
11. [参考来源](#11-参考来源)

---

## 1. 这套部署是什么

### 1.1 硬件与环境

| | head（rank 0） | worker（rank 1） |
|---|---|---|
| 管理网 | 192.168.1.32 | 192.168.1.33 |
| RoCE rail A | 192.168.100.1 | 192.168.100.2 |
| RoCE rail B | 192.168.101.1 | 192.168.101.2 |

- **NVIDIA GB10**（SM121，capability 12.1），128 GB 统一内存 × 2
- Ubuntu 24.04.4 aarch64，内核 6.17.0-1031-nvidia
- 驱动 580.173.02，CUDA 13.0，ConnectX-7 固件 28.45.4028
- **一根 QSFP 线，接在 Port 1**（两台之间）

### 1.2 软件栈

| | |
|---|---|
| 模型 | `deepseek-ai/DeepSeek-V4-Flash-0731`，155.43 GiB / 48 分片 |
| 运行时 | vLLM `0.21.1rc1.dev339+g1967a5627bc3` |
| 镜像 | `vllm-dspark-runtime:dspark-nvfp4-stage-c`（**必须含 Patch A**） |
| 容器 | `dsv4f-vllm-dspark-1`（两台同名） |
| 并行 | TP=2，PP=1，两节点 mp executor |
| 投机解码 | DSpark，k=5，`draft_sample_method: probabilistic` |
| KV cache | `nvfp4_ds_mla`，block-size 256 |
| 上下文 | 524,288 token |
| 服务端点 | `http://<head>:8078/v1`，模型名 `deepseek-v4-flash` |

### 1.3 生产配置四元组

这四个值是本部署相对上游模板的核心偏离，**报任何性能数字时都必须同时说明它们**：

```
gmu 0.8036 / cudagraph 记账 1 / fused-markov 1 / DRAFT_CAPTURE_SIZES auto
```

对应的环境变量：

```bash
GPU_MEMORY_UTILIZATION=0.8036
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1
VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto
```

一条命令核对全部（含两台一致性）：

```bash
./scripts/dsv4f-launch.sh --check-only
```

---

## 2. 先读这个：吞吐模型

**理解这套部署所有性能数字的钥匙只有一条公式：**

```
单流吞吐 = 步速 × 每步接受的 token 数
```

- **步速**几乎是常数，只跟并发有关：单流约 **13.5–14.5 步/秒**，c6 时约 **6.3–8.5 步/秒**（批次大，每步更慢）
- **每步接受 token 数**（下称"接受长度"）完全由**内容**决定，与并发无关

实测三档内容，同一台机器、同一次启动（2026-08-30；这里是**接受长度与吞吐的配对观测**，用于演示公式，与 [5.2 节的现行基线](#52-单流基线)不是同一组数据）：

| 内容类型 | 接受长度 | 单流 tok/s | 步速反算 |
|---|---|---|---|
| 数数（`Count from 1 to 300`） | 5.96 / 6 | 85.1 | 14.3 步/秒 |
| 代码生成 | 5.41 | 71.6 | — |
| 散文推理 | 2.36–2.53 | 35.9–37.5 | 15.2 / 14.8 步/秒 |

**单流吞吐从 36 到 85 tok/s，2.3 倍的差距全部来自接受率，GPU 步速纹丝不动。**

### 这条公式的三个推论

**① 一个吞吐数字不附带接受长度，就是不可解读的。**
看到"36 tok/s"你无法判断是系统慢了还是内容难预测。**报吞吐必须同时报接受长度。**

**② 估算新负载不需要重新压测。**

```
单流 tok/s ≈ 14.5 × 该内容的接受长度
c6 聚合   ≈ 单流 × 2.7
```

**③ 接受长度是最好的健康指标。**
投机解码由目标模型验证，**坏的 draft 只会让你变慢，永远不会让你变错**。所以接受长度下降 = drafter 出问题，而输出质量正常 —— 这个组合本身就是诊断结论。详见 [7.1](#71-核心风险能跑质量正常就是慢一半)。

---

## 3. 部署步骤

### 3.1 先修网络，再谈部署

**这一步不能跳过。** GB10 有一个已知故障：RDMA 带宽会被钳在 **13 Gb/s**（标称 200）。带宽坏了模型照样起、照样出字，只是慢 —— 所以它不会以任何显式错误暴露出来。

详细症状与排查过程见 [7.3](#74-网络-13-gbs-钳位)。这里只给结论和验收：

**修复方法**：接好最终布线 → 不再动线 → **重启两台** → 重测。

**验收（门槛 184 Gb/s）**：

```bash
# worker 上开两个终端
ib_write_bw -d rocep1s0f1   -x 3 -F --report_gbits -D 15 -p 18515
ib_write_bw -d roceP2p1s0f1 -x 3 -F --report_gbits -D 15 -p 18516

# head 上一条命令同时压两个域
( ib_write_bw -d rocep1s0f1   -x 3 -F --report_gbits -D 15 -p 18515 192.168.100.2 | awk '/^ [0-9]/{print "railA="$4}' & \
  ib_write_bw -d roceP2p1s0f1 -x 3 -F --report_gbits -D 15 -p 18516 192.168.101.2 | awk '/^ [0-9]/{print "railB="$4}' & \
  wait )
```

正常结果：**98.01 + 98.01 = 196.02 Gb/s**。

> **必须两个域同时压。** 一个物理 QSFP 口由两个独立 PCIe Gen5 x4 域各喂一半，每域上限约 126 Gb/s。**单跑一个 `ib_write_bw` 或 `iperf3` 永远只能看到约 98**，那不代表故障。

**网络配置持久化** —— `/etc/netplan/40-cx7.yaml`（head；worker 把末位改成 .2）：

```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    enp1s0f1np1:
      dhcp4: no
      addresses: [192.168.100.1/24]
      mtu: 9000
    enP2p1s0f1np1:
      dhcp4: no
      addresses: [192.168.101.1/24]
      mtu: 9000
```

> ### 运维纪律
> **任何时候动过 QSFP 线，必须重启两台并重跑双接口带宽测试。**
> 这个 bug 静默、可复现、代价 7.5 倍带宽。

### 3.2 配置文件

把 `config/env.dspark.example` 复制成部署目录的 `.env.dspark`，至少要改这些：

```bash
HF_TOKEN=<你的 HuggingFace token>
WORKER_HOST=192.168.100.2          # worker 的 RoCE 地址
MASTER_ADDR=192.168.100.1          # head 的 RoCE 地址
VLLM_HOST_IP=192.168.100.1
WORKER_VLLM_HOST_IP=192.168.100.2
WORKER_DIR=/home/<user>/services/dsv4f
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1,enP2p1s0f1np1
```

完整参数见 [第 4 节](#4-参数参考)。

### 3.3 打 Patch A

Patch A 给 DSpark drafter 一套独立于目标模型的 cudagraph 捕获尺寸。**实测单流 +6%，是整轮调优里唯一确凿的性能收益。**

**不要用 bind-mount。** 上游启动脚本的注释写得很清楚：

> DSpark source patches ship inside the runtime image (`recipe/overlay/`), **not as runtime bind-mounts**.

正确做法是覆盖 overlay 源文件，让构建链自己重建：

```bash
cp patches/dspark_proposer.py \
   ~/services/dsv4f/recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py
```

启动脚本的 overlay 过期检查会自动触发重建，**两台都建**：

```
verify-overlay-sources.sh → overlay 镜像 → stage-a → b → c
        ↓
build-dspark-vllm-runtime.sh 默认 WORKER_BUILD=1：
  rsync -az --delete 整个仓库到 worker → worker 同样重建一遍
```

整条链约 1 分钟/台。**两台一致性由设计保证，不需要手工同步。**

> ### ⚠️ 装了补丁还必须设开关，否则等于没装
>
> ```bash
> VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto
> ```
>
> 补丁自己的解析逻辑写着：`""/0/off -> None (feature off, byte-identical legacy behaviour)`。**不设这个变量，补丁的行为与原版逐字节相同。**
>
> 而且这个变量在上游的 `docker-compose.dspark.yml` 里**不存在**，只写进 `.env.dspark` 传不进容器。需要在 `environment:` 段加：
>
> ```yaml
> VLLM_DSPARK_DRAFT_CAPTURE_SIZES: "${VLLM_DSPARK_DRAFT_CAPTURE_SIZES:-}"
> ```
>
> 默认空即关闭，是个安全默认。

### 3.4 启动

```bash
# 1. 确认无残留容器（绝不 docker restart/start 旧容器）
docker ps -a | grep dspark

# 2. 清 page cache（两台，ssh 必须带 -t）
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
ssh -t <worker> "sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"

# 3. 启动（脚本自动 worker 先起、head 后起）
cd "$REPO_DIR" && ./dsv4f-launch.sh      # REPO_DIR 是部署目录，默认 ~/services/dsv4f
```

**冷启动约 8 分 30 秒**（首次含 JIT 编译；缓存建立后约 350 秒）。

### 3.5 验证：六项自检

`./scripts/dsv4f-launch.sh --check-only` 会查这些。**全绿才算部署成功** —— 每一项对应一个已知的静默失败：

| 检查 | 期望 | 失效的代价 |
|---|---|---|
| Patch 3（两台） | `grep -c is_prefill_chunk …/sched/scheduler.py` = **5** | 冷预填充 11/12 失败，热请求全过（冒烟测不出） |
| Patch 4（两台） | `grep -c shared_experts.gate_up_proj …/spec_decode/dspark.py` ≥ **2** | 接受率 60.2% → 25.7%，55.4 → 32.7 tok/s |
| B12X MoE（两台） | `VLLM_USE_B12X_MOE` = **1** | 静默回落 DEEPGEMM_MXFP4，55+ → 29 tok/s |
| proposer 两台 md5 一致 | 相同 | **两台代码不同会导致集合通信序列发散、NCCL 挂起** |
| Patch A 开关状态 | `auto` 时必须查到 `drafter-private cudagraph` 生效日志 | 补丁没进镜像 |
| 接受长度总闸 | **≥ 3.5** | 前三项任何一个失效都在它上面体现 |

> **md5 判据针对的是镜像内的文件，不是镜像 ID。** 两台各自 `docker build`，镜像 ID 天然不可复现；源码由 `rsync -az --delete` 强制同一份。用镜像 ID 比对会得到"不一致"的假警报。

自检还会回显配置四元组和当次日志建议的 gmu 值。

---

## 4. 参数参考

### 4.1 相对上游模板的偏离

| 参数 | 模板默认 | 本部署 | 依据 |
|---|---|---|---|
| `GPU_MEMORY_UTILIZATION` | 0.85 | **0.8036** | 上游竞赛实测 0.80 是物理边缘。**注意口径**：本部署 cudagraph 记账开着，0.8036 等效于记账关闭时的约 0.796，仍在边缘内。2026-09-02 从 0.7935 提上来，KV 池 +12.2%，单流/TTFT/正确性零变化，c6 压测无 OOM（[8.2](#82-gmu-与-cudagraph-记账必须一起改)）。0.85 是未跟上竞赛结论的遗留值 |
| `MAX_MODEL_LEN` | 1048576 | **524288** | 1M 上下文 TTFT 达 16 分钟，agent 场景不可用 |
| `MAX_NUM_SEQS` | 12 | **6** | 配合 512K 上下文留 KV 余量 |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 0 | **1** | cudagraph 内存计入 KV 分配，边缘不易 OOM。**必须同时上调 gmu**，否则 KV 池反而缩小（见 [8.2](#82-gmu-与-cudagraph-记账必须一起改)） |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX` | 0 | **1** | 避免物化词表大小的 Markov logits。2026-09-02 两侧各 5 轮实测**完全中性**（中位差 0.1%，KV 池差 0.5%），改回默认 0 是无损的 |
| `VLLM_DSPARK_DRAFT_CAPTURE_SIZES` | 不存在 | **auto** | Patch A 的开关，解析为 `[1,2,4,6]`。**单流 +6%** |

### 4.2 四个不能动的

| 项 | 原因 |
|---|---|
| `--block-size 256` | V4 FlashMLA + indexer 内核要求**恰好 256**，128/512 都不行 |
| `--generation-config vllm` | 不读模型自带的 `generation_config.json`。里面的 `repetition_penalty=1.05` 是有记录的 DSpark 崩溃风险（illegal memory access） |
| 不设 `--attention-backend` | 留 AUTO。`FLASHINFER_MLA_SPARSE_DSV4` 在本镜像不存在 |
| 不设 `VLLM_USE_V2_MODEL_RUNNER=1` | 与 DSpark 不兼容，启动即被拒 |

### 4.3 k=5 是硬约束

三条独立证据，**"降 k 求稳"是被明确证伪的做法**：

1. **上游竞赛实测**：k=4 电池均值 −6.1%（count −11.5%、tool −10%）。接受率虽升到 66%，但每步接受的 token 更少。k=3 因同机制"严格更差"被删除
2. **镜像硬限制**：`SpeculativeConfig.hf_config_override` 的 DSpark 分支把 `n_predict = dspark_block_size = 5`。k=7 启动即被除尽性检查拒绝；绕过后首次生成崩 `The size of tensor a (7) must match the size of tensor b (5)`。规则是 **k ≤ 5 或 5 的倍数**
3. **本机逐位接受率**：第 5 位仍有 0.737，明显赚钱

> DeepSeek 官方模型卡推荐 `num_speculative_tokens: 7`，但那是模型属性，不是本运行时的能力。

### 4.4 其余关键参数（沿用模板默认）

```bash
MTP_NUM_TOKENS=5
MAX_NUM_BATCHED_TOKENS=8192

# NCCL —— GID_INDEX 见下方说明
NCCL_IB_MERGE_NICS=1
NCCL_IB_GID_INDEX=3
NCCL_CROSS_NIC=1
NCCL_NET=IB
NCCL_CUMEM_ENABLE=0
NCCL_NVLS_ENABLE=0

VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_B12X_MOE=1                  # 关键：=0 静默掉到 29 tok/s
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0
VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
VLLM_DSPARK_LOCAL_ARGMAX=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
VLLM_DSV4_B12X_COMPRESSED_MLA=0
VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0

VLLM_ENGINE_READY_TIMEOUT_S=3600
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
TORCH_CUDA_ARCH_LIST=12.1a
FLASHINFER_CUDA_ARCH_LIST=12.1a
FLASHINFER_DISABLE_VERSION_CHECK=1
DG_JIT_USE_NVRTC=0
DG_JIT_NVCC_COMPILER=/opt/env/bin/nvcc
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> **两个死配置**：模板里的 `VLLM_TRITON_MLA_SPARSE` 和 `VLLM_SKIP_INIT_MEMORY_CHECK` 在整个 vLLM 包里**零引用**，设了不做任何事。留着无害，但别以为它们在控制什么。

**`NCCL_IB_GID_INDEX=3` 不能抄错**：3 = RoCE v2 + IPv4。idx 2 和 idx 3 的 GID 值完全相同、只有 type 不同，抄错会静默走 RoCE v1。验证：

```bash
cat /sys/class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/3   # 应输出 "RoCE v2"
```

### 4.5 完整 serve 命令行

```bash
dsv4-vllm-entrypoint serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 --port 8078 \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 524288 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.8036 \
  --enable-prefix-caching \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":false}' \
  --generation-config vllm \
  --enable-flashinfer-autotune \
  --nnodes 2 --node-rank ${NODE_RANK} \
  --master-addr 192.168.100.1 --master-port 25000 \
  ${HEADLESS:+--headless}
```

---

## 5. 性能基线与容量规划

### 5.1 四项标准测试

`scripts/bench-baseline.py` 用这四条提示词，**逐字固化在脚本里** —— 换了提示词就不再是同一个基准（接受率由内容驱动）。

| 测试名 | 提示词 | 代表的负载类型 |
|---|---|---|
| `count` | Count from 1 to 300, separated by commas. | 合成最好情况，高度可预测 |
| `struct` | Output a JSON array of 40 objects, each with fields id, name, email, active. | 结构化输出 |
| `code` | Write a Python function that implements a red-black tree with insert and delete. | 代码生成、工具调用参数 |
| `prose` | Write a 400-word essay on why distributed systems are hard. | 自由文本推理 |

### 5.2 单流基线

**生产配置，5×500 token 预热后，`stream:false`，t=0，3 次取中位数（2026-09-02）：**

| 测试 | 本机 | 三轮实测 | 离散度 | 上游参考 |
|---|---|---|---|---|
| count | **85.3** | 85.4 / 85.3 / 85.0 | 0.5% | 78.4 |
| struct | **77.9** | 77.9 / 77.7 / 78.9 | 1.5% | 66.1 |
| code | **74.1** | 72.1 / 74.1 / 75.7 | **5.0%** | 62.2 |
| prose | **36.7** | 36.0 / 37.3 / 36.7 | 3.6% | 37.8 |

> **`code` 这一项的三轮是单调递增的**（72.1 → 74.1 → 75.7），不像随机噪声，更像预热尚未到稳态。现行的 5×500 token 预热对它可能不够。**用这一项判断小效应时要当心** —— 优先用 count（离散度 0.5%）或 5.6 节的 TTFT（0.4%）。

**08-30 那组（85.5 / 74.6 / 71.6 / 36.8）已停用。** count 与 prose 在 09-02 复现得几乎完美（−0.2% / −0.3%），但 struct 与 code 稳定高出 4–6%。同一次测量里两项精确复现、两项一致偏高，不像噪声；成因未查，可能是 08-30 那次这两项本身偏低。

复现：

```bash
python3 scripts/bench-baseline.py warmup   # 5 轮 × ~500 token，到稳态
python3 scripts/bench-baseline.py bench    # 四项测试
```

**接受率（与上游对比）：**

| | 本机 | 上游参考 |
|---|---|---|
| 真实负载接受长度 | **5.19 / 6** | 4.0–5.0 |
| 真实负载接受率 | **83.8%** | 56.1% |
| 逐位接受率 | 0.937 / 0.874 / 0.853 / 0.789 / **0.737** | 0.826 / 0.725 / 0.572 / 0.471 / **0.399** |

每一位都明显高于上游，第 5 位 0.737 vs 0.399 —— 这是 k=5 仍然赚钱的直接证据。

### 5.3 并发容量

`scripts/loadtest.py`，同样是 3 轮中位数。**聚合吞吐 tok/s：**

| 内容类型 | 接受长度 | c1 | c2 | c4 | **c6** |
|---|---|---|---|---|---|
| count（合成最好情况） | 5.96 | 84.6 | — | 216.1 | **304.5** |
| **code**（工具调用/结构化/代码） | **5.41** | 73.0 | 107.9 | 153.9 | **200.3** |
| **散文**（自由文本推理） | **2.5** | 36.1 | 55.6 | 81.8 | **99.5** |

**单流中位（同批测量）：**

| 内容 | c1 | c2 | c4 | c6 |
|---|---|---|---|---|
| code | 73.0 | 54.5 | 39.5 | **34.2** |
| 散文 | 36.1 | 28.1 | 20.5 | **17.3** |

延迟（600 token 生成）：code c6 p50 **17.6s** / p95 **18.0s**。

**边际收益递减，c6 已接近饱和**：c1→c2 **+48%**、c2→c4 **+43%**、c4→c6 **+30%**。

> ### ⚠️ 不要测 c > 6
>
> `MAX_NUM_SEQS=6` 是引擎硬上限（日志实测 `Running: 6, Waiting: 2`）。超过 6 的并发测的是**突发投放的尾巴效应** —— 前 6 个按 c6 速度跑完，剩下的在低并发下单独跑，把墙钟拉长、聚合吞吐被稀释。
>
> 那个数字看起来像"并发越高越慢"，**但它不是引擎容量，是测试形态的产物**。`loadtest.py` 里加了守卫，`c > 6` 直接拒绝。

**不建议调高 `MAX_NUM_SEQS`**：`Maximum concurrency` 仅 2.44x（KV 池只够约 2.4 个满长度请求），调高会让长上下文更早触发抢占。**抢占的代价不是排队，是整段提示词重新预填充。**

### 5.4 与上游数据的交叉验证

上游那组容量数字（553 请求 / 21.3 万 token / 40 分钟）和本机实测**逐项吻合**：

| 上游记录 | 上游值 | 本机实测 | 差异 |
|---|---|---|---|
| c4 聚合（BST prompt） | 151.1 | code c4 **153.9** | +1.9% |
| c4 单流（BST prompt） | 38.7 | code c4 **39.5** | +2.1% |
| c4 聚合（真实混合流量） | 88.6 | 散文 c4 **81.8** | −7.7% |
| c4 单流（真实混合流量） | 22.3 | 散文 c4 **20.5** | −8.1% |

**两个结论：**

1. 上游的 "BST prompt" 基准就是 **code 类内容**，"真实混合流量"落在**散文档**附近 —— 两个口径现在都清楚了
2. **本机性能与上游同配置一致**，没有隐藏损失

### 5.5 怎么用这张表做容量规划

> **保守规划用散文档：c6 约 100 聚合 / 17 单流。乐观上限是 code 档的 200 / 34。**

真实 agent 负载里：

- **工具调用参数、结构化输出、代码生成** → 接受率接近 code 档（~5.4）
- **自然语言回复、推理链** → 接近散文档（~2.5）

上游"按 c4 约 88 聚合 / 22 单流规划"这条建议**仍然成立**，它对应的就是散文端。

### 5.6 预填充与 TTFT

2026-09-02 本机实测，`measure-prefill.py ttft`，每档 3 次取中位数，四元组 `gmu 0.7935 / 记账 1 / fused-markov 1 / auto`（当时的生产值；后于同日提到 0.8036，实测 TTFT 无变化）。

| prompt_tokens | TTFT 中位 | TTFT 范围 | 有效吞吐 |
|---|---|---|---|
| 768 | 0.56s | 0.54–0.65s | 1,371 tok/s |
| 6,101 | 3.17s | 3.16–3.27s | 1,925 |
| 24,361 | 11.98s | 11.91–12.01s | 2,033 |
| 76,068 | 38.88s | 38.82–38.99s | 1,956 |

**实用结论：76K token 的提示词，第一个字要等约 39 秒。**

**TTFT 是本机噪声最小的指标** —— 100K 档三次跨度 38.82–38.99，离散度 **0.4%**，远优于并发聚合的 ±11%。测小效应应当优先选它（第 8.7 节的稀疏索引器验收就是靠它做成的）。

#### 已证伪：预填充并不随深度变快

上游竞赛报的是 8K → 1,513、32K → 2,284、100K → **2,639** tok/s，即越长越快。**本机不成立**：8K 时我们更快（1,925 vs 1,513），100K 时明显更慢（1,956 vs 2,639），**8K 之后基本走平在约 2,000 tok/s**。

长上下文场景应当按**平坦的 ~2,000 tok/s** 来估算，不要指望深度带来加速。

#### 预填充由两段串行工作组成，引擎只统计了快的那段

引擎日志里的 `Avg prompt throughput` 每 10 秒打一次，100K 那档报 **7,606 tok/s** —— 而 `7,606 × 10s = 76,063`，正好是全部 prompt token。**所以这个数的含义是"整个预填充落在一个日志窗口内"，不是瞬时速率**，它对应的实际耗时约 10 秒。

但实测 TTFT 是 38.88 秒。把四档都按同样方式拆开：

| prompt_tok | 实测 TTFT | 引擎计入段 | 未计入段 | 未计入段速率 |
|---|---|---|---|---|
| 768 | 0.56s | 0.10s | 0.46s | 1,670 tok/s |
| 6,101 | 3.17s | 0.80s | 2.37s | 2,574 |
| 24,361 | 11.98s | 3.20s | 8.78s | 2,775 |
| 76,068 | 38.88s | 10.00s | 28.88s | 2,634 |

**未计入的那段也是线性的，三个长档稳定在约 2,660 tok/s。** 两段串联算回去 `1/(1/7606 + 1/2660) = 1,970 tok/s`，与实测的 1,956 吻合。

所以不存在"黑洞"，而是**两段串行的线性工作，慢的那段决定整体，且引擎的仪表看不见它**。

**这一段是什么，尚未确定。** 曾怀疑是 DSV4 的稀疏注意力索引器（`index_topk=512`、`index_n_heads=64`），但 2026-09-02 实测 `VLLM_USE_B12X_SPARSE_INDEXER=1` 让 TTFT 更慢（见第 8.7 节），**没有证实这个猜想**。次要嫌疑是 NVFP4 的 KV 量化写入，验证方法是临时换 `KV_CACHE_DTYPE=fp8_ds_mla` 对照。

#### 顺带：预热未覆盖的 JIT

长预填充时日志会出现：

```
WARNING [jit_monitor.py] Triton kernel JIT compilation during inference:
_build_prefill_chunk_metadata_kernel. This causes a latency spike;
consider extending warmup to cover this shape/config.
```

**但它不是上述开销的成因** —— JIT 是一次性的，而三轮 TTFT 离散度只有 0.4%（若是 JIT，首轮应明显更慢）。扩展预热形状可以消除首次请求的尖峰，收益有限。

#### 长预填充对短请求的阻塞

2026-09-02 实测，`measure-prefill.py blocking`。空闲时短请求 TTFT 中位 0.17–0.18s。

| 长请求在途 | 生产配置（阈值 1024）| 不设阈值时 |
|---|---|---|
| 6,102 tok | 1.54s（8.4×）| 2.90s（17.4×）|
| 24,360 tok | 1.59s（8.7×）| 11.42s（68.4×）|
| 76,068 tok | **1.69s（9.2×）** | 38.12s（**228.6×**）|

**不设阈值时，短请求几乎完整地等长请求预填充完** —— 76K 在途时要等 38.1s，而那条长请求自己的 TTFT 是 38.9s，重叠 98%。日志里能直接看到 `Running: 1 reqs, Waiting: 1 reqs`：短请求卡在等待队列，一次都没被调度进去。**分块预填充在这个场景下没有起到让路作用。**

而且阻塞**随长度超线性增长**（17× → 68× → 229×，token 只增长 12 倍），说明是完全串行。

**这一项直接推翻了第 8.4 节原先"不需要该参数"的结论**，详见该节的完整权衡。

### 5.7 KV 池：期望值与噪声

生产配置下 KV 池约 **1.21M–1.30M token**（gmu 0.8036，2026-09-02 共 7 次启动实测）。

> **2026-09-02 更新**：同配置 7 次启动实测跨度 **7.3%**（1,211,077 ↔ 1,299,486，中位 1,282,243）。比原先记的 11% 略窄。**判断任何 KV 池相关改动时，7% 以内的差异不构成证据** —— 同日 gmu 0.7935→0.8036 测出 +12.2%，正是因为高于这个范围才敢下结论。
>
> page cache 假说做了对照但**没能建立起来**：每次启动前必须先 `docker rm -f` 停容器，而停容器本身就释放了缓存，两组的启动前可用内存都是 116–117G，变量没被真正操纵。成因仍不明，但不影响使用 —— 已知波动范围，多次取中位数即可。

> ### ⚠️ KV 池的启动间波动可达 11%，单次测量不能用来评估配置改动
>
> 同配置多次启动实测：
>
> | 配置 | 各次 KV 池 | 跨度 |
> |---|---|---|
> | gmu 0.78 / 记账关 | 1,115,248 / 1,216,093 / 1,237,098 | **11%** |
> | gmu 0.7935 / 记账开 / Patch A on | 1,242,845 / 1,278,794 | **2.9%** |
>
> 曾据此得出过两个错误结论："Patch A 代价 KV −3.1%"（实际 ON 两次自己就差 2.9%）、"gmu 调优带来 +8%"（真实增益约 +3~4%）。
>
> **KV 池不会像吞吐那样瞬态跳变，但启动间波动同样需要多次采样。**

---

## 6. 测量纪律

这一节的每一条都是踩过之后才写下的。**不遵守会得到系统性错误的数字，而不是有噪声的数字。**

### 6.1 必须 `"stream": false`

投机解码下 vLLM 每个 decode **step** 最多发一个 SSE chunk，里面装着该步接受的**全部** token。

**数流式 delta 测的是 steps/s 而不是 tokens/s** —— 同一个请求 14.7 vs 60.1，**低报 4 倍**。

正确做法：`"stream": false`，读 `usage.completion_tokens`。

### 6.2 必须预热到稳态

**热身衰减是 30%，而且启动日志完全看不出来** —— 服务器报告 ready、cudagraph 已捕获、回答全程正确，就是慢。

上游实测：

```
刚启动（graph 已捕获、发过 3 次短预热）    58.5 tok/s
跑过约 5 次长生成之后                      83.3 / 83.2 / 83.1 / 83.2
```

**它还会衰减**：同一容器未重启，40 分钟压测后闲置约 30 分钟，count300 测得 **60.4**；重新大量预热后恢复 **83.5**。

**几次 100 token 的短预热不够，需要 500–700 token 级别的生成。** `bench-baseline.py warmup` 跑 5 轮 × ~500 token。

**对策：保温 cron**（`scripts/keepalive.sh`，每 15 分钟一次 600 token 生成，占空比约 0.8%）：

```
*/15 * * * * /home/<user>/services/dsv4f/keepalive.sh
```

要点：

- 检查 `vllm:num_requests_running`，非零就跳过测速（不抢并发槽），**但仍推监控心跳** —— 否则最忙的时候反而报故障
- **带 nonce 避开前缀缓存**，让预填充路径也保温、数字可比
- 同时记录 tok/s 和接受长度，日志本身就是健康曲线

### 6.3 压测提示词必须带 nonce

不加 nonce 时，同一长提示词第二次跑会命中前缀缓存，**墙钟从 5.58s 掉到 0.99s** —— 测的已经不是预填充路径了。

**nonce 必须放在提示词最前面**（前缀缓存按前缀匹配）。

### 6.4 任何性能结论至少 3 次取中位数

**本机会产生 2 倍的瞬态异常值。** 实测：count-300 一次报 **39.6 tok/s**（正常 82.6），复测两次回到 82.4 / 82.6。

排除项：同期 `Running: 1 / Waiting: 0`，**没有竞争流量**；接受长度 6.00（满分），**draft 质量正常**。所以不是负载也不是投机解码退化，是**每步耗时翻倍**的瞬态。

共同特征：都出现在**一段间隔后的第一个重负载请求**上。

> 各指标的噪声地板（本机实测）：
>
> | 指标 | 噪声 |
> |---|---|
> | 单流基线（3 次中位） | **±1%** |
> | 并发聚合 c6 | **±11%**，且偶发 2 倍异常 |
> | KV 池（启动间） | **±11%** |
>
> **效应量小于噪声地板时，那个实验无论跑多少次都只能产出"无差异"。** 选判据之前先知道它的噪声。

### 6.5 报数字时必须说明的三件事

1. **配置四元组** —— 否则读数的人不知道你测的是哪套配置
2. **接受长度** —— 否则无法判断是系统慢还是内容难
3. **温度** —— T>0 的惩罚是真实的：drafter 导出 one-hot 概率，散文类 t=0 约 40%、**T=0.7 只有 27.6%**。生产 agent 通常跑 T>0

---

## 7. 故障模式

### 7.1 核心风险：能跑、质量正常、就是慢一半

**这个部署的最大风险不是崩溃。** 投机解码由目标模型验证，坏的 draft 只会让你变慢、**永远不会让你变错**。

所以典型的故障长这样：服务正常启动、自检没报错、回答内容完全正确、**吞吐只有一半**。

> **"降速但质量无损"这个组合本身就是诊断结论** —— 它指向 drafter，而不是权重或配置。

**总闸指标：mean acceptance length ≥ 3.5**（健康 4.01–5.19，失效约 2.28）。

实时观察：

```bash
docker logs -f --tail=0 dsv4f-vllm-dspark-1 2>&1 \
  | grep --line-buffered -oE "Mean acceptance length: [0-9.]+|Avg generation throughput: [0-9.]+ tokens/s"
```

### 7.2 四个静默失败点

| # | 故障 | 症状 | 检查方式 |
|---|---|---|---|
| 1 | **Patch 3 未加载** | 冷预填充 11/12 失败，热请求 0/19 失败 | `docker exec <c> grep -c is_prefill_chunk …/sched/scheduler.py` → 5 |
| 2 | **Patch 4 未生效** | 接受率 60.2% → 25.7%，55.4 → 32.7 tok/s | `docker exec <c> grep -c shared_experts.gate_up_proj …/spec_decode/dspark.py` → ≥2 |
| 3 | **B12X MoE 未开** | 静默回落 DEEPGEMM_MXFP4，55+ → 29 tok/s | `docker exec <c> env \| grep VLLM_USE_B12X_MOE` → 1 |
| 4 | **Patch 5（stop 串）** | 客户端发 `stop` 时返回 `content: null`，答案静默丢失 | 见 [9.4](#94-开思考模式的前置条件) |

**Patch 3 的性质值得单独说**：它**只在冷预填充发作**，热请求全过 —— 冒烟测试永远测不出来。上游实测把 k 降到 3 也照样 10/10 失败，**`k` 不是那个变量，Patch 3 才是**。

### 7.3 检查方式本身的两个坑

这两个坑让上面的检查**给出稳定且错误的答案**，比没有检查更危险。

#### 坑 1：`grep -q` 放进 `pipefail` 的管道里

```bash
set -uo pipefail
docker logs "$CONTAINER" 2>&1 | grep -q "pattern"
```

**`grep -q` 一命中就立刻退出并关闭管道** → 上游 `docker logs` 收到 SIGPIPE 被杀（退出码 **141**）→ `pipefail` 判整条管道**失败**，尽管匹配是成功的。

实测：`pipefail` 下退出码 141，去掉 `pipefail` 后 0。

**后果是两个判据方向相反地坏掉：**

| 判据写法 | 实际行为 |
|---|---|
| `grep -q "警告串" && bad \|\| ok` | 管道永远失败 → **永远走 `ok`，永远静默通过**。这个检查等于从来不存在 |
| `grep -q "成功串" && ok \|\| bad` | **永远走 `bad`**，与日志在不在无关 |

**修法** —— 改用 `grep -c`（读完全部输入，不产生 SIGPIPE），count=0 时返回 1 故需 `|| true`：

```bash
n=$(docker logs "$CONTAINER" 2>&1 | grep -c "pattern" || true)
[ "${n:-0}" -gt 0 ] && ok "..." || warn "..."
```

> **上游输出越大越必然触发。** `docker logs`（几 MB）百分之百；`echo "$var" | grep -q` 这类小输出一次写满管道缓冲就退出，实际安全 —— 但没理由留着不改。
>
> **审查自检脚本时，把每个 `| grep -q` 都当成嫌疑。**

#### 坑 2：一次性日志会被环形缓冲覆盖

`Using 'B12X' Mxfp4 MoE backend` 这类启动期日志，跑几小时后就 grep 不到了。

**这是独立于坑 1 的第二个问题** —— 即使修好 SIGPIPE，长时间运行后日志判据仍会失效。

**所以前三个静默失败点都应该用 `docker exec` 直接查文件内容或环境变量，不依赖日志。**

### 7.4 网络 13 Gb/s 钳位

**症状**：RDMA 带宽钳在 13.41 / 13.36 Gb/s（标称 200），双 rail 合计 26。

**排查路径（全部无效，但值得记录）：**

| 假设 | 排除依据 |
|---|---|
| 网线接叉 | ARP 表验证同名口对同名口 |
| PCIe 降速 | `LnkSta: Speed 32GT/s, Width x4` 满速 |
| 拥塞控制 / 重传 / pause 帧 | hw_counters、ethtool 计数器全零 |
| SMMU 地址翻译 | 大页测试无改善 |
| 单 QP / 小消息开销 | **QP×8、消息×16 四组合，数字在 5% 内纹丝不动** |

延迟 `t_avg = 1.67 µs`（正常 2–5）。**延迟完美 + 带宽钳位 + 参数完全不敏感 = 钳位类问题，不是开销类问题。**

> **参数完全不敏感的性能问题不是调参问题。** 开销类问题一定随参数变化，钳位类不会。扫描曲线是水平线时应立即停止调参，转去查状态。

**根因**：GB10 已知的**初始化 / 热插拔状态**。**修复就是重启**（见 [3.1](#31-先修网络再谈部署)）。

**两个诊断陷阱：**

- `dmesg` 里的 `Detected insufficient power on the PCIe slot (27W)` 是**装饰性警告**，修好后照样打印，不能作为诊断依据
- 社区把这个修复归功于驱动升级（580.126 → 580.142），但本机 580.173 仍然故障 —— **真正起作用的是升级附带的那次重启**

**GB10 网络拓扑**（4 个接口 = 2 个物理口）：

| 接口 | PCIe 域 | RDMA 设备 | 物理口 |
|---|---|---|---|
| enp1s0f0np0 | 0000:01:00.0 | rocep1s0f0 | Port 0 |
| **enp1s0f1np1** | 0000:01:00.1 | **rocep1s0f1** | **Port 1** |
| enP2p1s0f0np0 | 0002:01:00.0 | roceP2p1s0f0 | Port 0 |
| **enP2p1s0f1np1** | 0002:01:00.1 | **roceP2p1s0f1** | **Port 1** |

> **注意**：上游竞赛实测双 HCA 对性能的影响是 **NULL** —— 互连不是这个负载的瓶颈（每步集合通信是延迟受限，预填充在 GB10 上计算受限）。修复 13 Gb/s 是从"坏"修到"够用"，不是从够用调到更好。

### 7.5 KV cache usage 高不是问题

开着 `--enable-prefix-caching` 时，已完成请求的 KV 块会留在池里当缓存，直到需要空间才驱逐。

**usage 95% 是正常的，甚至说明缓存在用。**

**真正的压力信号是 `Waiting` 非零和 preempt：**

```bash
docker logs --tail=300 dsv4f-vllm-dspark-1 2>&1 | grep -E "Waiting: [1-9]|Preempt"
```

抢占的代价不是排队，而是**整段提示词重新预填充**。

---

## 8. 调优记录：试过什么，结论是什么

### 8.1 总表

| 改动 | 判决 | 实际价值 |
|---|---|---|
| **Patch A** + `DRAFT_CAPTURE_SIZES=auto` | ✅ 保留 | **单流 +6%**，唯一确凿的性能收益 |
| `ESTIMATE_CUDAGRAPHS=1` + gmu 0.7935 | ✅ 保留 | 换的是**正确性**（cudagraph 内存计入分配，边缘不易 OOM），KV +3~4% |
| `FUSED_MARKOV_ARGMAX=1` | ⚪ 中性 | 2026-09-02 两侧各 5 轮严格 A/B：中位 85.0 vs 85.1（差 **0.1%**），KV 池差 0.5%。**上游报的 +1.8% 在本机不成立，省显存带宽一说也无实测支撑。** 保留现状而非改回默认，只因改动无收益也无损失 |
| 预填充调度两参数 | ❌ 证伪 | V1 上一个是死参数，另一个语义与名字相反 |
| cudagraph 捕获尺寸收窄 | ❌ 证伪 | 完全中性，收益与代价都在噪声之下 |

**四项改动里真正带来性能的只有 Patch A。** 另外两项买的是正确性和理论余量。

### 8.2 gmu 与 cudagraph 记账必须一起改

**启动日志建议的 gmu 值取决于它是在记账开还是关的状态下打印的**，两者算的不是同一笔账：

| 当次配置 | 日志说"等效于" | 日志建议改成 |
|---|---|---|
| 0.7800 / 记账关 | — | 0.7873 |
| 0.7873 / 记账开 | 0.7811 | 0.7935 |
| 0.7935 / 记账开 | 0.7860 | 0.8010 |

**照着"记账关闭"时的建议（0.7873）去开启记账，会让 KV 池缩小 5.4%。**

正确做法：**先开记账启动一次，再照它新打印的建议值调。**

固定偏移约 **0.006–0.0075**：记账开着的 gmu 减去这个数，才是关闭口径下的等效值。所以 **0.7935（开）≈ 0.7860（关）**，仍低于竞赛实测的 0.80 物理边缘。

> **自动建议不是常量，是当前状态的函数。改了状态就要重新取一次建议。**

### 8.3 Patch A：唯一确凿的收益

**决定性证据是单流基线**（并发扫描的噪声太大，掩盖了真实效果）。规范提示词各测三次，离散度约 1%：

| 测试 | `off` | `auto` | 差异 |
|---|---|---|---|
| count | 81.0 | **85.7** | **+5.8%** |
| struct | 70.7 | **75.0** | **+6.1%** |

上游报 +5% c4，本机 c4 实测 +2.3~6.2%，**数量级吻合**。

> **教训**：先前用并发扫描（±11%、偶发 2 倍异常）评估它，得出"中性偏混合"，差点因此放弃。**用噪声大的指标测小效应会得到假阴性。**

**实验设计上有个便宜的地方值得学**：补丁声称 `off -> byte-identical legacy behaviour`。这意味着**不用为对照组重建镜像** —— 同一个镜像、同一份代码，只翻一个环境变量，就得到教科书级的 A/B。

**而且这个承诺本身值得验证**：设 `off` 后生效日志消失、KV 池回到原值，说明关闭路径确实干净。如果数字**没**回来，那才是更要紧的发现 —— 说明整个对照实验的前提垮了。

**副产品**：目标侧捕获尺寸从 12 档砍到 7 档时，drafter 的 `[1,2,4,6]` 纹丝不动，日志同步报告 `target sizes stay [...]`。`_DrafterCompilationConfigView` 确实做到了它声称的隔离。

### 8.4 预填充调度两参数：一个是死的，一个是生产配置

```
--max-num-partial-prefills 1        # 死参数，不要设
--long-prefill-token-threshold 1024 # 已采纳，2026-09-02 起进入生产
```

抄自上游 sparkrun port 时，两条都被理解为"限制长预填充同时只有一个"。读源码加实测之后，**两条的结论完全不同**。

#### ① `max_num_partial_prefills` 在 V1 上是死参数

`grep -rn max_num_partial_prefills .../vllm/v1/` **零命中** —— 只存在于 V0 代码路径，而本部署跑 V1（启动日志 `Initializing a V1 LLM engine`）。默认值本来就是 1，双重无效。**这条判断从头到尾正确。**

#### ② `long_prefill_token_threshold` 是活的，而且默认永远不生效

它在 V1 调度器里被真实使用（`v1/core/sched/scheduler.py:390`）：

```python
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
```

**每步每请求的 token 硬上限**，不是并发限制 —— 参数名描述的是意图，不是行为。

**但它的自动赋值路径被前面那个死参数堵住了**（`config/scheduler.py:244`）：

```python
if self.max_num_partial_prefills > 1:          # V1 上永远不成立
    if self.long_prefill_token_threshold == 0:
        self.long_prefill_token_threshold = int(max_model_len * 0.04)
```

所以**不显式传参，它永远是 0（不生效）**。这解释了为什么它长期没被真正评估过：从外部行为看两个参数都"不起作用"，但成因不同 —— 一个是**代码不存在**，另一个是**代码存在但入口被堵死**。

#### 实测权衡（2026-09-02，阈值 1024）

**收益：阻塞几乎消失，而且不再随长度增长**

| 长请求在途 | 不设 | 设 1024 | 改善 |
|---|---|---|---|
| 6,102 tok | 2.90s（17.4×）| 1.54s（8.4×）| −47% |
| 24,360 tok | 11.42s（68.4×）| 1.59s（8.7×）| −86% |
| 76,068 tok | 38.12s（**228.6×**）| **1.69s（9.2×）** | **−96%** |

（倍数相对空闲时短请求 TTFT 0.17–0.18s）

**关键不是数值降低，是阻塞从"随长度爆炸"变成了常数。** 不设时 17× → 68× → 229×；设了之后 8.4× → 8.7× → 9.2× —— 文档大 12 倍，等待只多 10%。

**代价：长请求 TTFT +28%**

| prompt_tok | 不设 | 设 1024 | 变化 |
|---|---|---|---|
| 767 | 0.56s | 0.56s | 0% |
| 6,102 | 3.17s | 3.82s | +21% |
| 24,361 | 11.98s | 15.30s | +28% |
| 76,068 | 38.88s | 49.66s | **+28%** |

**单流基线不受影响**：84.8 / 77.9 / 74.0 / 36.5，四项全部在噪声内。**纯单流场景零代价，纯粹是并发行为的改变。**

#### 结论：采纳

牺牲一个本来就要等 39 秒的请求 11 秒，换另一个请求从 38 秒降到 1.7 秒。对 agent 混合负载这个交换很划算。

**若你的负载永远是单流，则没有收益，只有 28% 的损失 —— 不要设。**

#### 副产品：引擎的仪表因此变准了

不设阈值时，`Avg prompt throughput` 在 100K 档报 7606 tok/s，但那其实是"整段预填充挤进一个 10 秒日志窗口"的产物（见 [5.6](#56-预填充与-ttft)）。设 1024 之后预填充摊平到多个窗口，四个长度档都稳定报 **7683 tok/s** —— 这个读数现在是真的速率了。

#### 怎么启用

上游的 compose 没有这个参数，需要自己加。本仓库 `config/env.dspark.example` 里给了 `LONG_PREFILL_THRESHOLD`，在 compose 的 `vllm serve` 参数段加一行：

```yaml
        --enable-chunked-prefill
        ${LONG_PREFILL_THRESHOLD:+--long-prefill-token-threshold ${LONG_PREFILL_THRESHOLD}}
```

写成可选形式，**变量为空时完全不传，行为与不加改动时一致**。两台 compose 都要改（启动脚本会 rsync，但改动要在 head 侧先做）。

验证方法是看启动日志的 `non-default args` 里有没有 `'long_prefill_token_threshold': 1024`。

> **参数名描述的是意图，不是行为。** 决定要不要用一个参数之前，找到它被读取的那一行 —— 并且确认那一行**真的会被执行到**。这一节两次栽在这上面：第一次以为它是并发限制，第二次以为它不起作用。

### 8.5 已证伪：收窄 cudagraph 捕获尺寸

**动机看起来很扎实**：vLLM 默认

```
max_cudagraph_capture_size = min(max_num_seqs × (1+k) × 2, 512) = min(6×6×2, 512) = 72
```

其中 `×2` 是通用余量。而 `--max-num-seqs 6` 硬性封顶并发序列（压测日志实测 `Running` 从未超过 6），**纯 decode 批次上限 = 6×6 = 36**，所以 40/48/56/64/72 五个图永远用不到。

设 `--max-cudagraph-capture-size 36` 后实测：

| 指标 | 默认 72 | 收窄到 36 |
|---|---|---|
| 捕获列表 | `[1,2,4,8,16,24,32,40,48,56,64,72]` | `[1,2,4,8,16,24,32]` |
| Graph capturing | 0.74 GiB / 9 秒 | **0.40 GiB / 4 秒** |
| KV 池 | 1,278,794 | 1,279,421（**无变化**） |

**结论：完全中性 —— 收益和代价都在噪声之下。**

**为什么省下的 0.34 GiB 没变成 KV**：预期增益 ≈ 2.6%，而 KV 池的启动波动是 2.9%（[5.7](#57-kv-池期望值与噪声)）。效应量小于噪声地板。

**参数值本身也设错了**：vLLM 生成的档位是 `[…24, 32, 40…]`，设上限 36 后列表停在 **32**，而 c6 满载批次是 **36** —— 卡在两档之间，没有图覆盖，回落 eager。

> **设"上限"类参数前，先确认这个值落在系统生成的档位上。** 上限参数通常不是连续的，而是用来截断一个离散序列。要覆盖 36 得设 **40**。

若日后仍想试，设 40 只砍掉确定用不到的 48/56/64/72。但收益仍在噪声之下 —— **测不出来就等于不存在，不值得为此携带一个未经证实的偏离**。

`docker-compose.dspark.yml` 里保留了可选参数形式，变量留空则完全不传：

```yaml
${MAX_CUDAGRAPH_CAPTURE_SIZE:+--max-cudagraph-capture-size ${MAX_CUDAGRAPH_CAPTURE_SIZE}}
```

### 8.6 明确不做

> `long_prefill_token_threshold` 曾长期列在本表里。2026-09-02 实测后**已移出并进入生产配置**（短请求阻塞 −96%，代价长请求 TTFT +28%），见 [8.4](#84-预填充调度两参数一个是死的一个是生产配置)。同组的 `max_num_partial_prefills` 仍留在表内。

| 项 | 理由 |
|---|---|
| k 改成 4 或 3 | 三条独立证据钉死 k=5（[4.3](#43-k5-是硬约束)） |
| gmu 上到 0.80 / 0.85 | 竞赛实测 0.80 已在物理边缘。**注意口径**：0.7935 是记账**开**的值，等效关闭口径 0.7860 |
| `max_num_partial_prefills` | 已证伪，V1 上零引用（[8.4](#84-预填充调度两参数一个是死的一个是生产配置)） |
| 收窄 cudagraph 捕获尺寸 | 已证伪（[8.5](#85-已证伪收窄-cudagraph-捕获尺寸)） |
| 拉到 1M 上下文 | 客户端侧也是 524288，服务端拉大零收益；sparse MLA 还会多分配依赖最大长度的工作区 |
| `VLLM_USE_B12X_MHC=1` | 竞赛 E9 启动即崩（`Can't export tensors that require gradient`） |
| `VLLM_DSV4_B12X_COMPRESSED_MLA=1` | 竞赛 E12 **输出错误** + CUDA assert |
| `VLLM_USE_B12X_FP8_GEMM=1` | 上游文档警告：DSpark drafter 预热时触发 DeepGEMM layout assertion |
| 双 HCA 调优 | 竞赛 E6 实测为 null，互连不是瓶颈 |
| `--max-num-batched-tokens 16384` | 竞赛 E7 撞 KV 悬崖 |
| MLA sparse 三个细分参数 | 预填充计算受限，±2% 噪声内 |
| `--block-size` 128 或 512 | 内核要求恰好 256 |

> **B12X 系列有"能跑但算错"的先例**（`COMPRESSED_MLA`）。这个系列的任何开关，验收都必须比对输出正确性，不能只看速度。

### 8.7 未测的候选

`VLLM_USE_B12X_SPARSE_INDEXER=1` —— B12X 系列里唯一既没有警告记录、上游竞赛也没测过的。所有门槛都过：`is_device_capability_family(120)` 对 SM121 返回 True，`use_fp4_indexer_cache` 默认 False。

**验收必须比对输出正确性。**

---

## 9. 客户端集成注意事项

本节讲的是**服务端 API 本身的行为** —— 任何客户端接入时都会遇到，与你用哪个框架无关。

### 9.1 响应字段是 `reasoning`，不是 `reasoning_content`

**这个运行时上没有 `reasoning_content` 这个 key** —— 已废弃，只在*输入*时接受。

读 `reasoning_content` 的客户端永远看到空值，然后得出"reasoning 提取坏了"的结论。流式下同理，delta 里是 `delta.reasoning`。

### 9.2 `reasoning_effort` 会真的开启思考模式

实测（`stream:false`，直连服务端）：

| 请求 | `reasoning` 字段 | completion_tokens |
|---|---|---|
| 不传 `reasoning_effort` | **空** | 224 |
| `reasoning_effort: "medium"` | **满**（结构化 CoT） | **398** |
| `reasoning_effort: "high"` | **满**（简短 CoT） | 77 |

**它等效于 `chat_template_kwargs: {"thinking": true}`。**

**代价买单两次**：同一个问题 `medium` 用掉 398 token 而不传只要 224，多出来的全是散文类 CoT —— 而散文接受率只有 25–33%，**这些 token 还跑得特别慢**。

> 这一条曾被记成"设 high 大概率什么也不会发生"，**方向正好相反**。当初测错是因为只对比了 `content` —— 而它在开关两侧确实都干净。**全部变化发生在 `reasoning` 字段里，而那个字段因为 9.1 的字段名问题永远读到空。**
>
> **判断一个开关是否生效前，先确认你正在看的字段真的是它会改变的那个。**

### 9.3 服务端解析器是好的 —— 泄露都在客户端

四种组合实测，**CoT 从未进入 `content`**：

| stream | thinking | `reasoning` 字段 | `content` |
|---|---|---|---|
| false | false | 空 | 干净 |
| false | true | 完整 CoT | 干净 |
| true | false | 空 | 干净 |
| true | true | 136 个 delta | 干净 |

**所以如果客户端的回复里混进了思维链，问题不在服务端。** 按这个顺序查：

1. **源头** —— 发出的请求里有没有 `reasoning_effort`（解析 JSON 看 `request.body` 的 key，**不要 grep 文件**：系统提示词里可能有说明文字，grep 会误报）
2. **中间层** —— 如果客户端和服务端之间有代理或网关，检查它有没有"把 reasoning 合并进 content"的选项。**这类开关一旦打开，现象和服务端泄露一模一样，而服务端完全无辜**（例如 LiteLLM 的 `merge_reasoning_content_in_choices`）
3. **客户端的显示开关** —— 同一个框架里可能有多个独立开关控制思维链显示，改一个不影响另外几个；还可能出现"配置写了但代码从来不读"的情况（配置层级写错，加载函数读的是另一个路径）。这类失效不会有任何报错，只能靠读源码发现
4. **最后才怀疑服务端解析器**

**从源头关掉，比在显示开关里猜哪个管用要干净得多。**

### 9.4 开思考模式的前置条件

两条等效路径：`chat_template_kwargs: {"thinking":true}` 或 `reasoning_effort`，任一条都够。开之前确认：

1. **客户端读的是 `reasoning` 字段**（9.1）
2. **Patch 5 变成必需** —— 思考模式下生成从 `<think>` 内部开始，思维链会复述提示词短语，客户端的 `stop` 提前触发 → `</think>` 永不出现 → **`content: null`，答案静默丢失**
3. **速度会掉到散文档**（约 35–40 tok/s），token 消耗显著上升（224 → 398）
4. **中间层和客户端显示开关都要确认**（9.3）

---

## 10. 方法论

这 26 条是这套部署过程中反复付出代价才总结出来的。前面各节引用的原则都在这里。

### 关于测量

**1. 参数完全不敏感的性能问题不是调参问题。**
QP×8、消息×16、大页、jumbo 四个变量扫完，数字在 5% 内纹丝不动 —— 开销类问题一定随参数变化，钳位类问题不会。**扫描曲线是水平线时应立即停止调参，转去查状态。**

**2. 先用噪声地板定义"算数"的门槛。**
把 baseline 原样重跑一次定义噪声，任何结果必须超过地板才计入。否则会把噪声当收益。

**3. 本机会产生 2 倍的瞬态异常值 —— 单次测量不能作为结论。**
count-300 一次报 39.6（正常 82.6），复测两次回到 82.4 / 82.6。同期无竞争流量、接受长度满分。**差点据此把一个无害改动误判成 −52% 回归并回滚。** 任何性能结论至少 3 次取中位数。

**4. 用噪声大的指标测小效应，会得到假阴性。**
先用并发扫描（±11%、偶发 2 倍异常）评估 Patch A，结论是"中性偏混合"，差点放弃。改用单流基线（±1%）后测出确凿的 +6%。**选判据之前先知道它的噪声地板** —— 效应量若小于地板，那个实验无论跑多少次都只能产出"无差异"。

**5. 一个吞吐数字不附带接受长度，就是不可解读的。**
同一台机器、同一次启动，单流从 36 到 85 tok/s，**2.3 倍差距全部来自内容的接受率**，而 GPU 步速几乎恒定。推论：估算新负载不需要重新压测，**单流 tok/s ≈ 14.5 × 该内容的接受长度**。

**6. 换了提示词就不再是同一个基准。**
要做趋势对比，**提示词必须逐字固化并存进仓库**，否则每次"基线"都是新基线。

**7. "和历史数字对不上"有两种可能，不重测对照组就无法区分。**
某项基线比历史值低 11%，看起来像回归。把变量全部回退后重测，发现原始配置下**同样是那个数** —— 历史值才是不可复现的。而且改动其实让它涨了 7.3%。**在拿到同日同配置的对照数据之前，"我退步了"和"那个历史数字不可靠"是无法分辨的。**

**8. 标称值和可达值是两回事。**
GB10 上标称 200G 的口，每个 PCIe 域封顶约 126 Gb/s，实测天花板 196。

**9. 比较不同 recipe，先看它优化的是什么指标。**
四节点无投机 256 并发 vs 双节点 k=5 单流，比较绝对数字没有意义。

### 关于配置与参数

**10. 抄参数前先确认它在你这条代码路径上还活着。**
`max_num_partial_prefills` 在 V1 代码里 grep 零命中 —— 它是 V0 遗留。参数被接受、不报错、写进命令行、`docker compose config` 也渲染得好好的，**但引擎从头到尾没读过它**。这类"配置存在但代码不读"的失效不会有任何报错，只能靠读源码发现。

**11. 参数名描述的是意图，不是行为。**
`long_prefill_token_threshold` 听起来是"长预填充的判定阈值"，实际是"每步每请求的 token 硬上限"。按名字理解会得出完全相反的收益判断。**决定要不要用一个参数之前，找到它被读取的那一行。**

**12. 设"上限"类参数前，先确认这个值落在系统生成的档位上。**
按"实际上限 36"设 `--max-cudagraph-capture-size 36`，但系统生成的档位是 `[…24, 32, 40…]` —— 列表停在 32，满载批次反而失去了 cudagraph。**上限参数通常不是连续的，而是用来截断一个离散序列。**

**13. 软件给的建议值，要问它是在哪个状态下算出来的。**
启动日志建议 gmu 提到 0.7873，照做之后 KV 池反而**缩小 5.4%** —— 因为那条建议是在记账关闭时打印的，而我们同时把记账打开了。**自动建议不是常量，是当前状态的函数；改了状态就要重新取一次建议。**

**14. 模板默认值可能落后于同仓库的实测结果。**
`.env.dspark.example` 写 `gmu=0.85`，而同仓库五天前的竞赛结论是 0.78；sparkrun port 写 `k=3`，而 README 明说那是 bug 值。**竞赛结果只写进了文档，没同步回模板。**

**15. 社区数字要看它跑在哪个版本上。**
有人把 13 Gb/s 的修复归功于驱动升级，但本机更高版本仍然故障 —— **真正起作用的是升级附带的那次重启**。

### 关于检查与监控

**16. 每个监控项都要问一遍"什么情况下它会说谎"。**
保温脚本忙碌时跳过测速 → 不推心跳 → 最忙时报故障。**监控逻辑本身有盲区比没有监控更危险** —— 它给你一种被覆盖了的错觉。

**17. `grep -q` 放进 `pipefail` 的管道里，会把"匹配成功"变成"退出码失败"。**
`grep -q` 命中即退出并关闭管道，上游被 SIGPIPE 杀掉（141），`pipefail` 于是判整条管道失败。**这类检查不会报错，只会给出稳定且错误的答案** —— 一个永远通过，一个永远失败，取决于你把 `&&` 还是 `||` 接在后面。**审查自检脚本时，把每个 `| grep -q` 都当成嫌疑。**

**18. 一次性日志不能作为长期判据。**
环形缓冲会覆盖启动期日志。要查配置是否生效，用 `docker exec` 读文件或环境变量。

**19. `grep` 到字符串不等于参数生效。**
session dump 里搜得到 `reasoning_effort`，但那是系统提示词里的说明文字。必须解析 JSON 看 `request.body` 的 key。**在结构化数据里做子串匹配，得到的是"提到过"，不是"用上了"。**

**20. 只对比"看得见的字段"会得出方向相反的结论。**
`reasoning_effort` 被判为"什么也不会发生"，是因为对比的是 `content` —— 全部差异在 `reasoning` 字段里。**判断一个开关是否生效前，先确认你正在看的字段真的是它会改变的那个。**

### 关于变更与协作

**21. 装补丁之前，先找仓库自己的补丁机制。**
用仓库自带的机制，两台一致性、过期检查、构建后冒烟测试全都免费拿到；自己发明 bind-mount 则要手工保证这些，还会绕过过期检查。**自制的挂载不会报错，只会在几个月后变成一个查不出的诡异 bug。**

**22. 声称"关掉即等价"的特性开关，是一个免费的对照组。**
同一个镜像、同一份代码，只翻一个环境变量，就得到教科书级的 A/B。**而且这个承诺本身值得验证** —— 如果关掉后数字没回来，那才是更要紧的发现，说明整个对照实验的前提垮了。

**23. 拓扑记漏一层，整条排查链都错位。**
中间那层代理带着能改写请求/响应的开关。**按错误的拓扑排查，会把中间层的行为归因到两端。** 画链路图时每一跳都要有实证。

**24. 用 heredoc 写配置文件，终端截断会静默产生半个文件。**
`EOF` 变成内容、变量没展开，都不报错。拆成 `cp` + 若干条 `sed` 就没这个问题 —— 每条命令要么成功要么明确失败，而且 `diff` 能精确告诉你改了什么。

**25. 验证方式必须覆盖改动的类型 —— 语法检查挡不住语义错误。**
用正则批量改脚本后只跑了语法检查，结果是：import 被删掉、旧函数还在用那些名字，**一运行就 NameError**，而语法完全合法。同一次改动还让文档描述了一个代码里不存在的行为（"参考基线为空则不打印偏差列"，实际会 KeyError）。
**改代码就要跑代码。** 服务不可用时至少确认它跑到了真正的工作点（比如网络调用）才失败，而不是在导入或取名字时就崩。**"能跑到网络调用"仍然不等于"能跑完"** —— 这两者之间还有整个响应解析路径。

**26. 排查"时好时坏"之前，先数一遍到底跑着几个实例。**
部署时曾有一个跑了近 5 天的旧客户端实例没人知道，配置停在迁移前，请求的模型名服务端直接返回 404 —— 它没造成可见故障纯粹是运气。**同一个服务的第二个实例是最难想到的变量**，因为所有人的心智模型里它只有一个。

---

## 11. 参考来源

- **主线 recipe**：[tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)（MIT）
  - 仓库内 `RESULTS-BLUEY-2026-08-20.md` —— 16 次启动的调参竞赛，本文多处引用的"竞赛 E6/E7/E9/E12"出自此处
  - 仓库内 `AGENT_GARBLE_FIX.md` —— Patch 3 与冷启动乱码
  - 仓库内 `DSPARK-SHARED-EXPERT-FIX.md` —— Patch 4
- **DSpark 并发补丁**：Keys / drowzeys，见 [CREDITS.md](../CREDITS.md)
- NVIDIA 开发者论坛 363461 —— **13 Gb/s 的真正修复（重启）**
- Chronara《GX10 ConnectX-7: Why You're Getting 13 Gbps》 —— GB10 PCIe 拓扑
- vLLM issues #51318 / #52836 / #52492 —— 乱码相关

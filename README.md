**中文** · [English](README.en.md)

# DeepSeek-V4-Flash-0731 · 双 GX10 部署

两台 NVIDIA GB10（DGX Spark / GX10）上跑 DeepSeek-V4-Flash-0731，TP=2、DSpark 投机解码 k=5、NVFP4 KV cache。

这个仓库是**运行记录**，不是教程：配置、自检脚本、测量工具，以及一批被实测推翻的结论。所有数字均本机实测。

> ## ⚠️ 这里测的是设备的极限，不是模型的质量
>
> **本仓库不含任何能力评测** —— 没有 benchmark 分数、没有输出质量对比、没有"这个模型好不好用"的结论。
>
> 全部内容只回答一件事：**这套硬件加这套配置能撑住多少、在哪里到顶、什么情况下会静默地慢一半。**
>
> 这决定了很多看起来奇怪的选择。比如四条标准测试提示词（count / struct / code / prose）是按**接受率档位**挑的，不是按能力维度挑的 —— 它们的作用是覆盖从"高度可预测"到"自由文本"的整个范围，好让吞吐数字可解释（见[吞吐模型](docs/deployment-notes.md#2-先读这个吞吐模型)）。**拿它们评价模型能力没有意义。**
>
> 同理，文中反复出现的噪声地板、多次取中位数、静默失败排查，都是为了让**设备侧的数字可信**，与模型输出的好坏无关。

---

## 现行生产配置

```
gmu 0.8036 / cudagraph 记账 1 / fused-markov 1 / DRAFT_CAPTURE_SIZES auto
```

| | |
|---|---|
| 服务端点 | `http://<head>:8078/v1`，模型名 `deepseek-v4-flash` |
| 硬件 | 2 × NVIDIA GB10（SM121），128GB 统一内存，CX7 双 rail RoCE |
| vLLM | `0.21.1rc1.dev339+g1967a5627bc3` |
| 权重 | `deepseek-ai/DeepSeek-V4-Flash-0731`，155.43 GiB / 48 分片 |
| 镜像 | `vllm-dspark-runtime:dspark-nvfp4-stage-c`，**必须含 Patch A** |

一条命令核对全部：

```bash
./scripts/dsv4f-launch.sh --check-only
```

## 性能基线（2026-08-30）

规范提示词，5×500 token 预热后，`stream:false`，t=0，3 次中位数。

| 内容 | 单流 tok/s | 接受长度 |
|---|---|---|
| count | 85.3 | 5.96 |
| struct | 77.9 | — |
| code | 74.1 | 5.41 |
| prose | 36.7 | 2.5 |

并发聚合吞吐（`MAX_NUM_SEQS=6` 是硬上限）：

| 内容 | c1 | c2 | c4 | c6 |
|---|---|---|---|---|
| code | 73.0 | 107.9 | 153.9 | **200.3** |
| prose | 36.1 | 55.6 | 81.8 | **99.5** |

> **报吞吐必须同时报接受长度。** 同一台机器单流从 36 到 85 tok/s，2.3 倍差距全部来自内容的接受率，而 GPU 步速几乎恒定（约 14.5 步/秒）。
>
> 估算新负载：**单流 tok/s ≈ 14.5 × 接受长度**，c6 聚合 ≈ 单流 × 2.7。

## 前置要求

| | |
|---|---|
| 硬件 | **两台** NVIDIA GB10（DGX Spark / GX10），128 GB 统一内存，CX7 双 rail RoCE 互连 |
| 系统 | Ubuntu 24.04 aarch64，驱动 580.x，CUDA 13.0 |
| 软件 | Docker + Compose、`bash`、`python3`（脚本只用标准库，无需 pip 安装） |
| 网络 | **RDMA 带宽必须先验收到 184 Gb/s 以上** —— 见笔记第 3.1 节，这一步不能跳过 |
| 凭据 | HuggingFace token（拉取模型权重，155.43 GiB） |
| 上游 | 本仓库是[上游 recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) 的下游部署记录，需要先有那套 recipe 的构建环境 |

## 仓库结构

```
docs/deployment-notes.md  完整部署笔记（配置、故障排查、容量规划、26 条方法论）
scripts/
  dsv4f-launch.sh          前置检查 → 启动 → 六项自检 → 可选基线
  bench-baseline.py        单流基线，提示词固化保证跨次可比
  loadtest.py              并发压测 c1/2/4/6，带 nonce 避开前缀缓存
  keepalive.sh             保温 cron，对抗热身衰减
  measure-prefill.py       TTFT vs 提示词长度；长预填充对短请求的阻塞
  _common.py               共享配置与工具（换模型只改这里的输入）
  profiles/<name>.json     每个模型一份：端点、并发上限、是否投机、参考基线
patches/
  dspark_proposer.py       Patch A（drafter 专属 cudagraph 尺寸，Apache-2.0）
  README.md                 安装说明与实测数据
config/
  env.dspark.example       生产配置模板（token 已抽掉）
TODO.md                    待完成的测量与证据强度清单
CHANGELOG.md               配置与工具的变更记录
NOTICE                     逐文件的许可归属
```

### ⚠️ 这个仓库不是部署目录

**本仓库放的是记录和工具，不是可以直接启动服务的部署目录。** 真正的部署在别处（上游 recipe 的 checkout），里面有 `start-deepseek-v4-flash-dspark.sh`、`.env.dspark`、`recipe/overlay/` 等本仓库不含的东西。

`dsv4f-launch.sh` 会 `cd` 到部署目录去调用那些文件，路径由 `REPO_DIR` 决定：

```bash
# 默认假设部署在 ~/services/dsv4f
./scripts/dsv4f-launch.sh --check-only

# 部署在别处时
REPO_DIR=/opt/dsv4f ./scripts/dsv4f-launch.sh --check-only
```

`dsv4f-launch.sh` 与 `keepalive.sh` 可配置的环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `REPO_DIR` | `~/services/dsv4f` | **部署目录**（不是本仓库） |
| `WORKER` | `192.168.100.2` | worker 的 RoCE 地址，自检要 ssh 过去 |
| `CONTAINER` | `dsv4f-vllm-dspark-1` | 容器名 |
| `PORT` / `MODEL` | `8078` / `deepseek-v4-flash` | 服务端点与模型名 |

> **网络相关的 `HCA_A` / `IP_A` / `HCA_B` / `IP_B` 是硬编码在脚本里的**，因为它们和本部署的双 rail 布局绑定。换网络布局需要直接改脚本开头那几行。

### 换个模型用这套工具

测量脚本不含硬编码，配置从 profile 文件或环境变量读：

```bash
# 用现成 profile
PROFILE=dsv4f ./scripts/bench-baseline.py bench

# 或直接用环境变量（非投机模型、并发上限 16）
ENDPOINT=http://host:8000 MODEL=qwen3-32b MAX_NUM_SEQS=16 SPECULATIVE=0   ./scripts/loadtest.py sweep
```

新模型复制 `scripts/profiles/dsv4f.json` 改值即可。仓库里另有 `glm53.json` 作对照 —— 那是同一套工具指向另一个模型（不同配方、不同并发上限、不同投机机制），可以看出哪些项是必改的。可配置项：端点、模型名、
容器名（读引擎日志用）、`max_num_seqs`（并发守卫上限）、`speculative`
（非投机模型会跳过接受长度采集）、参考基线（为空则不打印偏差列）。

加 `--md` 输出 markdown 表格，直接粘进笔记。

> ### ⚠️ 这四个 Python 工具尚未在真实服务上验证
>
> 它们共用 `_common.py` 里的请求与解析路径，而这套代码只经过语法、配置加载和并发守卫的测试——**没有一个完整跑通过一次真实请求**。第一次用它们时请参照 [TODO 第 0 项](TODO.md#0-先验证工具本身--已完成2026-09-02)，那里列了建议的验证顺序和具体未验证的点。

## 快速上手

1. 把 `config/env.dspark.example` 复制成部署目录的 `.env.dspark`，填入你自己的 `HF_TOKEN` 和两台的 IP
2. 把 `patches/dspark_proposer.py` 覆盖到 `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`，跑启动脚本会自动重建两台镜像
3. **在 `.env.dspark` 里设 `VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto`** —— 不设的话补丁装了等于没装，见 [patches/README.md](patches/README.md)
4. `./scripts/dsv4f-launch.sh` —— 六项自检全绿才算部署成功

## 三个最容易踩的坑

**1. 静默失败比崩溃危险。** 核心风险不是起不来，而是"能跑、输出质量正常、就是慢一半"。投机解码由目标模型验证，坏的 draft 只会让你变慢、永远不会让你变错 —— **"降速但质量无损"这个组合本身就是诊断结论**。总闸是 mean acceptance length ≥ 3.5。

**2. `grep -q` 放进 `pipefail` 的管道里会撒谎。** `grep -q` 命中即退出关闭管道，上游 `docker logs` 被 SIGPIPE 杀掉（141），`set -o pipefail` 于是判整条管道失败。**这让自检给出稳定且错误的答案** —— 一个永远通过，一个永远失败。改用 `grep -c`。

**3. 单次测量不能作为结论。** 本机会产生 **2 倍的瞬态异常值**（count-300 一次报 39.6，正常 82.6，复测两次回到 82.4/82.6）。KV 池的启动间波动可达 11%。任何性能结论至少 3 次取中位数。

完整的 26 条见 [部署笔记](docs/deployment-notes.md#10-方法论)。

## 被实测推翻的结论

这份记录的一大半价值在于**推翻自己**。主要几条：

| 曾经的结论 | 实测 |
|---|---|
| 客户端直连服务端 | 中间隔着一层代理，它带着能改写请求/响应的开关 |
| `reasoning_effort` 设了没用 | 方向相反 —— 它就是思考模式的开关，token 消耗 224 → 398 |
| B12X 误报是日志被覆盖 | 真因是 `grep -q` + `pipefail` 的 SIGPIPE，与日志在不在无关 |
| struct 基线掉了 11% | 不是回归，是历史参考值不可复现；调优其实让它涨了 7.3% |
| 抄 sparkrun 的两个预填充参数 | 一个在 V1 上是死参数，另一个语义与名字相反 |
| 收窄 cudagraph 捕获尺寸能省显存 | 完全中性，收益与代价都在噪声之下 |
| 两台镜像 ID 不同需要统一 | 伪问题 —— 判据是镜像内文件 md5，不是镜像 ID |

## 许可与归属

**本仓库许可不统一，逐文件说明见 [NOTICE](NOTICE)。** 概要：

| 内容 | 许可 |
|---|---|
| 笔记、`bench-baseline.py`、`loadtest.py`、`patches/README.md` | MIT，本仓库原创 |
| `dsv4f-launch.sh`、`keepalive.sh`、`env.dspark.example` | MIT，衍生自[上游 recipe](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) |
| **`patches/dspark_proposer.py`** | **Apache-2.0** —— vLLM 衍生，文件头保留原始 SPDX 标识 |

模型权重、基础镜像、CUDA / NCCL / FlashInfer / TileLang / Triton 不在本仓库内，各有自己的许可条款。

**本仓库是上游 recipe 的下游部署记录，不是它的替代品。** DSpark 并发工作来自 Keys / drowzeys。完整归属见 [CREDITS.md](CREDITS.md)。

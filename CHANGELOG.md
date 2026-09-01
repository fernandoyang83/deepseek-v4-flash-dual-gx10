**中文** · [English](CHANGELOG.en.md)

# Changelog

本文件记录部署配置与工具的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

日期为实测日期，所有性能数字的测量方法见 `docs/deployment-notes.md` 第 6 节。

---

## [1.0.0] — 2026-09-01

首次发布。DeepSeek-V4-Flash-0731 在两台 NVIDIA GB10（DGX Spark / GX10）上的部署记录：配置、自检、测量工具，以及一批被实测推翻的结论。

### 包含

- **部署笔记**（`docs/deployment-notes.md`）—— 完整部署路径、参数真值、容量基线、故障排查，以及 26 条方法论。
- **生产配置** —— gmu 0.7935 / cudagraph 记账 1 / fused-markov 1 / DRAFT_CAPTURE_SIZES auto。
- **Patch A**（`patches/dspark_proposer.py`）—— drafter 专属 cudagraph 尺寸，单流基线 **+6%**（count 81.0 → 85.7、struct 70.7 → 75.0，三次离散度约 1%）。这是整轮调优中唯一确凿的性能收益。**必须同时设 `VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto`，否则补丁装了等于没装。**
- **启动脚本**（`scripts/dsv4f-launch.sh`）—— 预检 → 启动 → 六项自检 → 可选基准。自检含 proposer 两台 md5 一致性与 Patch A 开关状态。
- **测量工具** —— 单流基线、并发压测、TTFT/预填充、保温 cron。配置从 profile 文件或环境变量读，可指向其他模型。
- **待办清单**（`TODO.md`）—— 六项未完成的测量，以及一张现有结论的证据强度表。

### 部署过程中被实测推翻的结论

这些是笔记的主要价值，完整记录见第 8 节：

- **自检脚本的 `grep -q` + `pipefail` bug** —— `grep -q` 命中即关闭管道，上游 `docker logs` 被 SIGPIPE 杀掉（退出码 141），`pipefail` 判整条管道失败。后果是一项判据**永远静默通过**、另一项**永远误报**，取决于后面挂的是 `&&` 还是 `||`。改用 `grep -c`。
- **预填充调度两参数证伪** —— `max_num_partial_prefills` 在 V1 代码路径上零引用，是 V0 遗留；`long_prefill_token_threshold` 的行为与名字相反，它是每步每请求的 token 硬上限，是延迟换吞吐的权衡而非免费优化。
- **收窄 cudagraph 捕获尺寸完全中性** —— 收益与代价都落在噪声地板之下，回退实验验证过。
- **"两台镜像 ID 不同需要统一"是伪问题** —— 两台各自 `docker build`，ID 天然不可复现；判据是镜像内文件的 md5，不是镜像 ID。
- **KV 池不是确定性的** —— 同配置启动间波动可达 11%，此前据此得出的两个结论因而作废。
- **客户端拓扑漏记了中间的代理层** —— 该层带着能把 `reasoning` 合并进 `content` 的开关，一旦打开，现象和服务端泄露一模一样。
- **`reasoning_effort` 的结论方向相反** —— 它就是思考模式的开关（token 消耗 224 → 398）。当初测错是因为只对比了 `content`，而变化全在 `reasoning` 字段里。

### 已知限制

- **第 5.6 节的预填充与 TTFT 数据是上游的，本机未测** —— 见 [TODO 第 1 项](TODO.md)。
- **四个 Python 测量工具尚未在真实服务上完整跑通** —— 见 [TODO 第 0 项](TODO.md)。

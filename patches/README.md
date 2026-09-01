**中文** · [English](README.en.md)

# Patch A —— drafter 专属 cudagraph 尺寸

`dspark_proposer.py` 是 vLLM `v1/spec_decode/dspark_proposer.py` 的替换版本，比原文件多 133 行、少 2 行。

## 它做什么

给 DSpark drafter 一套**独立于目标模型**的 cudagraph 捕获尺寸，按**请求数**索引而不是目标 token 数。启用后日志会打印：

```
DSpark drafter-private cudagraph capture sizes enabled: [1, 2, 4, 6]
(target sizes stay [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72]).
Draft graphs are keyed by request count; a batch of B requests now runs B x 5 draft tokens.
```

## 实测收益

单流基线，规范提示词，各测三次（离散度约 1%）：

| 测试 | `off` | `auto` | 差异 |
|---|---|---|---|
| count | 81.0 | **85.7** | **+5.8%** |
| struct | 70.7 | **75.0** | **+6.1%** |

**这是整轮调优里最确凿的单项收益。** 上游报 +5% c4，本机 c4 实测 +2.3~6.2%，数量级吻合。

> 先前用并发扫描评估它，得出"中性偏混合"，差点因此放弃。**并发聚合指标的噪声 ±5% 且偶发 2 倍异常，盖过了这个 +6%。** 用噪声大的指标测小效应会得到假阴性 —— 单流基线（±1%）才分辨得出来。

## 安装

**不要用 bind-mount。** 启动脚本的注释写得很清楚：

> DSpark source patches ship inside the runtime image (`recipe/overlay/`), **not as runtime bind-mounts**.

正确做法是覆盖 overlay 源文件，让构建链自己重建：

```bash
cp patches/dspark_proposer.py \
   ~/services/dsv4f/recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py

cd ~/services/dsv4f && ./dsv4f-launch.sh
```

启动脚本的 overlay 过期检查会自动触发：

```
verify-overlay-sources.sh → overlay 镜像 → stage-a → b → c
        ↓
build-dspark-vllm-runtime.sh 默认 WORKER_BUILD=1：
  rsync -az --delete 整个仓库到 worker → worker 同样重建一遍
```

**两台一致性由设计保证，不需要手工同步。** 整条链约 1 分钟/台。

## ⚠️ 装了补丁还必须设开关

```bash
VLLM_DSPARK_DRAFT_CAPTURE_SIZES=auto
```

**不设这个变量，补丁装了等于没装。** 补丁自己写着：

```
""/0/off  ->  None (feature off, byte-identical legacy behaviour)
```

而且这个变量原先不在 `docker-compose.dspark.yml` 的 `environment:` 列表里，只写进 `.env.dspark` 传不进容器。需要加：

```yaml
VLLM_DSPARK_DRAFT_CAPTURE_SIZES: "${VLLM_DSPARK_DRAFT_CAPTURE_SIZES:-}"
```

默认空即关闭，是个安全默认。

## 验证

`dsv4f-launch.sh --check-only` 会查两项：

- **两台镜像内 `dspark_proposer.py` 的 md5 一致** —— NCCL 挂起的直接守卫。判据是**文件 md5，不是镜像 ID**（两台各自 build，ID 天然不同）
- **`auto` 时必须能查到生效日志**，否则报错

## 副产品：隔离是干净的

目标侧捕获尺寸从 12 档砍到 7 档时，drafter 的 `[1,2,4,6]` 纹丝不动，日志同步报告 `target sizes stay [...]`。`_DrafterCompilationConfigView` 确实做到了它声称的隔离。

"关掉即等价"这个承诺也验证过：设 `off` 后生效日志消失、KV 池回到原值。**这意味着可以用同一个镜像做干净的 A/B，不必为对照组重建。**

## 归属

补丁来自上游 [tonyd2wild](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)（MIT）。DSpark 并发工作来自 Keys / drowzeys，见根目录 [CREDITS.md](../CREDITS.md)。

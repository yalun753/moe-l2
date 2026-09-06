# moe-l2

[English](README.md) | [**中文**](README_zh.md)

[![PyPI version](https://img.shields.io/pypi/v/moe-l2)](https://pypi.org/project/moe-l2/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![CI](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml/badge.svg)](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml)

**MoE 专家卸载（expert offload）低显存方案 — 10-11GB 显卡也能跑 100B+ MoE 大模型（DeepSeek、Qwen、Mixtral），省 57-60% 显存，一行 pip 搞定。**

> ⭐ **觉得有用？点个 Star** —— 让更多需要的人发现它。[★ 去 GitHub 点赞](https://github.com/yalun753/moe-l2)

> 💬 **实测数据汇总 & 快速上手** —— 完整 benchmark 表格、内存数字、安装指南：[Discussions #2](https://github.com/yalun753/moe-l2/discussions/2)

| 你的显卡 | 正常能跑 | **用了 moe-l2** | **实测速度**（RTX 4090） |
|----------|---------|-----------------|----------------------|
| 10-11 GB | — | DeepSeek-V2-Lite (16B MoE) ✅ | **139-154 t/s** |
| **10-11 GB** | 7B 稠密模型 | **Qwen3.6-A3B (32B MoE) ✅** | **25.5-44.2 t/s** |
| 10-11 GB | — | **DeepSeek-V4-Flash（157B MoE，85 GB 文件）⚠️** | **N/A — 上游 llama.cpp deepseek4 CUDA 专家 bug（[#25582](https://github.com/ggml-org/llama.cpp/issues/25582)）** |
| 24 GB | — | **Qwen3-235B-A22B（235B MoE，85.7 GB 文件）✅** | **~3.9 t/s** |

> 速度 = RTX 4090 实测（2026-08-19，bins-v0.6.0 per-slot 锁版，全链路 `moe-l2 start --gpu`）：DS-V2-Lite 139-154 t/s（单轮 141、长文 139）、Qwen3.6-A3B 25.5-44.2 t/s（混合领域）；2080 Ti 全链路（v0.6.0）：Qwen 16.6-28.6 t/s、DS-V2-Lite 86-94 t/s。**5090（SM120a，v0.6.0）：Qwen 28-52.5 t/s、DS 141-151 t/s**。Qwen3-235B-A22B：稳态 ~3.9 t/s（2026-08-11）。详见 [models-benchmark.md](references/zh/models-benchmark.md)。

**DeepSeek-V4-Flash（157B 参数 / 85GB 文件，256 专家、激活 6）也能跑**——显存 16.5-16.7GB（on-demand 兜底，RSS 17.5GB；selective pin v4_top100.map：RSS 26.8GB），2026-08-10 实测。⚠️ **无有效速度数据——上游 llama.cpp bug（[#25582](https://github.com/ggml-org/llama.cpp/issues/25582)）**：deepseek4 MoE 专家层在 CUDA 上输出乱码（纯原版 llama-server 复现，与 moe-l2 无关）；UD-IQ2_M / Q4_K_XL 均受影响，等上游修复。 完整报告：[deepseek-v4-flash-verify-20260805.md](references/zh/deepseek-v4-flash-verify-20260805.md) · **Qwen3-235B-A22B（235B 参数 / 85.7GB 文件，128 专家、激活 8）在 24GB 卡上也能跑**——RTX 4090 实测稳态 **~3.9 t/s**（selective pin top-60/层，覆盖 98.5%）：24GB 显存 + 55GB 内存，RSS 80.8 → 54.7GB（-33%），2026-08-11 实测。完整报告：[qwen3-235b-a22b-q2k-benchmark.md](references/zh/qwen3-235b-a22b-q2k-benchmark.md) · **全部已测模型汇总：[models-benchmark.md](references/zh/models-benchmark.md)**

### 可视化演示（RTX 4090，2026-08-16）

| Qwen3.6-35B-A3B（32B MoE）— 标准 vs moe-l2 | DeepSeek-V2-Lite（16B MoE）— 8GB 卡 vs 24GB 卡 |
|---|---|
| ![Qwen 显存对比](examples/demo-assets/fig1-qwen-vram.png) | ![DS 显存对比](examples/demo-assets/fig2-ds-vram.png) |

一句话总结：**显存省 57-60%** —— 10-11 GB 卡跑出原本 24 GB 卡的效果（RTX 4090 实测 2026-08-19，bins-v0.6.0：DS 139-154 t/s @ ~10.1 GB / Qwen 25.5-44.2 t/s @ ~6.7-7.2 GB）：

![moe-l2 汇总](examples/demo-assets/fig3-summary.png)

实机录屏（2026-08-16，bins-v0.5.0 C 方案）：Qwen3.6-35B-A3B 生成 **2344 tokens，显存 ~5.4 GB**（~32 t/s）——显存曲线全程平稳：

[`examples/demo-assets/demo-vram-animation.mp4`](examples/demo-assets/demo-vram-animation.mp4)（45 秒，1280×720）· 原始采样：[`examples/demo-assets/rec_data.csv`](examples/demo-assets/rec_data.csv) · 生成全文：[`examples/demo-assets/rec_full.txt`](examples/demo-assets/rec_full.txt)

---

### Benchmarked on RTX 4090（2026-08-19，v0.6.0）

基于 **RTX 4090** 实测（2026-08-19，selective pin + 按领域换表主路径：路由表按领域驱动 top-K pin + 换表 + GPU cache 预填充，bins-v0.6.0）：

| 模式 | GPU 显存 | 速度 | 意味着什么 |
|------|----------|------|-----------|
| 标准（全 expert 在 GPU） | 23.3 GB | 65 t/s | 需要 24 GB 显卡 |
| **moe-l2**（selective pin 专家，GPU 计算） | **~6.7-10.1 GB** | **DS 139-154 t/s · Qwen 25.5-44.2 t/s** | **10-11 GB 卡也能跑** |
| **节省** | **57-60% 显存** | ~205% 全 GPU 速度 | 腾出 ~14 GB 做别的 |

不开 moe-l2，**10 GB 以下显卡根本无法加载这个模型**——直接 OOM。开了之后 32B MoE 占 ~9.3 GB（selective pin 专家，GPU 计算）。

> 我们在 RTX 4090 上对 **Qwen3.6-A3B**（32B MoE）和 **DeepSeek-V2-Lite**（16B MoE，64 expert）做了全量测试（2026-08-19，selective pin + 按领域换表 + A3 cache，bins-v0.6.0 per-slot 锁版）：专家驻留 CPU RAM（零显存），路由表按领域预选高频专家，调度器每步只把**激活的专家**拷到 GPU 直算，热专家缓存在 GPU 显存。DS-V2-Lite **139-154 t/s**（~10.1GB 显存、~6.4-6.7GB RSS），Qwen3.6-A3B **25.5-44.2 t/s**（混合领域；~6.7-7.2GB 显存、~8.9-11.3GB RSS）。完整报告：[Qwen3.6](references/zh/qwen3.6-a3b-iq2m-benchmark.md) · [DS-V2-Lite](references/zh/deepseek-v2-lite-q2k-benchmark.md) · [models-benchmark](references/zh/models-benchmark.md)

### Selective pin — 低内存模式（V4 RSS 对比，2026-08-10 实测）

![Selective pin RSS 对比——whole-pin 84GB vs selective pin 26.8GB vs on-demand 17.5GB，DeepSeek-V4-Flash UD-IQ2_M @ RTX 4090](docs/demo/fig5-selective-pin-rss.png)

*实测（RTX 4090，2026-08-10，bins-v0.4.0）：whole-pin 84GB → selective pin 26.8GB（路由表 top-K）→ on-demand 兜底 17.5GB。RSS 降 68%（V4 内存数据有效；速度 N/A — 上游 deepseek4 CUDA bug [#25582](https://github.com/ggml-org/llama.cpp/issues/25582)）。另见 [速度 vs 内存散点图](docs/demo/fig5b-selective-pin-speed-rss.png)。*

**Selective pin（低内存模式，v0.4.0 起；当前主路径为 v0.6.0 按领域换表）**——路由表（每层 top-K 专家，如 `v4_top100.map` 43 层）预 pin 高频专家为 host-pinned，表外专家走 on-demand 兜底。不设环境变量时保持 whole-pin 默认；`moe-l2 start --gpu` 传 `--router-map <文件>` 或 `--router-top-k N`：

```bash
moe-l2 start --model model.gguf --gpu --router-map v4_top100.map
```

### 多架构二进制（bins-v0.8.0，2026-09-06）

**一个二进制兼容所有 NVIDIA 显卡**——GTX 1080（sm_61）/ P100（sm_60）到 RTX 50 系（sm_120a），并从 bins-v0.8.0 起支持原生 Windows（`llama_bins_win.zip`）。CUDA 12.8 编译，无需按显卡单独编译，`moe-l2 download-bins` 自动拉取（PyPI 0.12.0 起按平台自动选 zip/tar.gz）。bins-v0.8.0 在 v0.7.0 引擎基础上新增 **sm_60 + sm_70**（P100 / V100 / Titan V）并附 **Windows zip**（llama-server.exe + DLLs，AVX2，内置 CUDA 12.8 runtime，驱动 ≥ 570）。v0.7.0 的功能全部继承：**IQ1_M 量化修复**（Qwen3.8-Flash-Next / Qwen4exp 125B 512 专家模型不再在 MMQ 崩溃——改走 MMVQ + 批上限拆分）+ **NCCL 多卡支持恢复**（打包内置 `libnccl.so.2`，`--split-mode layer/row/tensor` 恢复可用）+ selective pin（路由表驱动）+ GPU cache 预填充 + on-demand pin 主路径 + 专家页淘汰 v3.1 + 分层 pin + A3 cache 32768 槽 + 按领域动态换表（POST /moe-set-domain）+ per-slot 锁优化 + soft_resize / 保留热专家 v2（默认单表）+ 显存自适应主表 top-k + proxy 并发修复 + flywheel 表持久化。

| GPU | 架构 | DS-V2-Lite | Qwen3.6-A3B | Qwen4exp 125B IQ1_M | 显存 |
|-----|------|-----------|-------------|---------------------|------|
| RTX 2080 Ti | sm_75（Turing） | 86-94 t/s | 16.6-28.6 t/s | — | Qwen 5.3 GB / DS 10.0 GB |
| RTX 3080 Ti | sm_86（Ampere） | 12.25 t/s | 13.28 t/s | — | ~1.1-2.2 GB |
| RTX 4090* | sm_89（Ada） | 139-154 t/s | 25.5-56.3 t/s | **19.8 t/s** | Qwen 6.7-7.2 GB / DS 10.1 GB |
| RTX 5090 | sm_120a | **141-151 t/s** | **28-52.5 t/s** | — | Qwen 5.6 GB / DS 10.2 GB |

\* 全部为全链路实测（`moe-l2 start --gpu`，按领域换表 + A3 cache，默认单表）。**bins-v0.7.0（2026-08-28）**：4090——Qwen4exp 125B IQ1_M **19.8 t/s**（cache hit 97.3%，32768 槽）、Qwen3.6 **56.3 t/s**（第 3 轮，比 v0.6.0 的 44-48 快 17-28%）、DS-V2-Lite **143.5 t/s**（+8%）；2080 Ti——Qwen3.6 **40.3 t/s**（hit 95.1%）。旧行：4090 v0.6.0 Qwen 25.5-44.2（混合领域）/ DS 139-154；2080 Ti v0.6.0 Qwen 16.6-28.6 / DS 86-94；5090 v0.6.0 Qwen 28-52.5 / DS 141-151；3080 Ti 仍为 v3.1 多架构包（bins-v0.3.0）实测。全部输出验证无乱码。**Qwen4exp 125B**——512 专家/层、IQ1_M：此前必崩（MMQ quantize 不支持 IQ1_M → GGML_ABORT）；修复 = 改走 MMVQ + 批上限拆分。三模型 4090 日志：`测试数据备份/4090-bins-v070-verify-20260828/`。

### 并发请求 — 共享 cache，速度不掉（2026-08-12）

4 路并发共享同一份 A3 专家 cache / selective pin 路由表——在 2080 Ti 和 4090 上用 **Qwen3.6-35B-A3B**、**DS-V2-Lite**、**DeepSeek-V4-Flash（256 专家、路由分散）** 验证：

| 模型（GPU） | 单会话 | 4 路并发·同领域 | 4 路并发·跨领域 | vs 单会话 |
|---|---|---|---|---|
| Qwen3.6-35B-A3B（2080 Ti） | 38.4 t/s | **95.02** 总（23.76×4） | **88.28** 总（21.7-22.2×4） | 2.3-2.5× |
| DS-V2-Lite（2080 Ti） | 78.3 t/s | **198.59** 总 | **188.25** 总 | 2.4-2.5× |
| DeepSeek-V4-Flash（4090）⚠️ | 35.4-35.8 t/s | **89.66** 总 | **88.10** 总 | 2.5× |

> ⚠️ V4 行速度 N/A — 上游 deepseek4 CUDA 专家路径 bug（#25582）。

并发总吞吐 = 单会话的 **2.3-2.5 倍**；跨领域 vs 同领域只差 **5-7%**——**无需按领域分 cache 池**。显存只随 slot 数线性增加（4 路 +2.9GB 全是 KV cache），内存几乎不动（+0.2GB）。**一台 AI PC 可以多人同时用。** 完整报告：[concurrent-cache-sharing-20260812.md](references/zh/concurrent-cache-sharing-20260812.md)

## 快速开始

### 1. 安装

```bash
pip install moe-l2
```

### 2. 下载 GPU 二进制（仅 `--gpu` 模式需要）

```bash
moe-l2 download-bins
```
从 GitHub Release 拉取预编译的 CUDA llama-server（bins-v0.6.0，约 1.6 GB 多架构全兼容包，含 cuda-libs）。

### 3. 启动

**GPU 模式（推荐，on-demand pin 专家 GPU 直算，省 57-60% 显存）：**
```bash
moe-l2 start --model /path/to/model.gguf --gpu
```

**纯代理模式**（不省显存，仅 expert 缓存）：
```bash
moe-l2 start --model /path/to/model.gguf --l2-size 4GB
```

代理启动在 `localhost:11435`，直接当普通 OpenAI 兼容接口用就行（GPU 模式下后端 llama-server 监听 11436）。

### 4. 看统计

```bash
moe-l2 stats
# → 命中率: 85% · 槽位: 320/960 · 当前领域: codegen
```

---

## 工作原理（简版）

1. 你的 prompt 到达 moe-l2 代理
2. 领域预测器分类（代码生成 → 数学 → 中文技术 ……）
3. 专家权重 lazy mmap 驻留 CPU RAM（**零显存**），首次激活即 pinned（on-demand pin）——不再整体塞进 GPU
4. GPU 经 PCIe DMA 直读激活专家（cuBLAS），热专家驻留 VRAM（A3 LRU 2048 槽），冷页 v3.1 淘汰
5. 可选 sched-cache（`GGML_CUDA_EXPERT_CACHE=0.25`）：命中热专家走 D2D 免 PCIe，DS 类模型 Prompt +211%

---

## 适用场景

### ✅ 适合
- **单人聊天** — 用 4-12 GB 显卡跑大 MoE 模型
- **测试和实验** — 在预算硬件上玩 MoE 架构
- **家庭实验室 / 边缘部署** — 每一 GB 显存都珍贵
- **研究 expert 缓存**、分层调度、领域感知预加载

### ❌ 不适合
- 高并发 API 服务（频繁换 expert 产生 I/O 瓶颈）
- 对延迟敏感的应用（SSD 缓存缺失导致速度波动）
- 机械硬盘作存储 — **需要 NVMe 固态**

---

## 系统架构

```
                         ┌─────────────────────────┐
   你的 prompt ──────────▶│  moe-l2 代理 (:11435)    │
                         │                          │
                         │  ┌─────────────────────┐ │
                         │  │ 领域预测器           │ │
                         │  │ (关键词+TF-IDF+语义) │ │
                         │  └────────┬────────────┘ │
                         │           │ 预测领域       │
                         │           ▼               │
                         │  ┌─────────────────────┐ │
                         │  │ L2 缓存 (RAM)       │ │
                         │  │ LRU · mmap · 异步   │ │
                         │  │ 从硬盘预加载         │ │
                         │  └────────┬────────────┘ │
                         │           │ 转发           │
                         └───────────┼───────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │  llama-server (:11436)   │
                         │  on-demand pin 专家：    │
                         │  lazy mmap 零显存，首次  │
                         │  触碰即 pinned，GPU 经   │
                         │  PCIe DMA 直读；热专家   │
                         │  驻留 VRAM (A3 LRU)，   │
                         │  冷页淘汰 RSS 封顶       │
                         └─────────────────────────┘
```

### 数据驻留层级（2026-08-07 架构）

```
L0 ─ CPU 路由器    门控路由 + 领域分类（你的 CPU）
 ↑
L1 ─ GPU 显存      激活 expert 计算 + KV cache（你的显卡，只放激活权重）
 ↑
L2 ─ CPU RAM       全部专家权重 lazy mmap 驻留，首次触碰即 pinned（零显存）
 ↑
L3 ─ SSD 冷存储    GGUF 文件 mmap，冷专家页按需读入 + v3.1 淘汰（RSS 封顶）
```

专家权重整体 lazy mmap 驻留 CPU RAM（零显存），首次激活即 pinned，GPU 经 PCIe DMA 直读；热专家驻留 VRAM（A3 LRU 2048 槽），冷页 v3.1 淘汰保持 RSS 封顶——这就是为什么 32B MoE 只需 ~9.3 GB 显存（标准全量 23.3GB）、85GB V4 也能在 11GB 卡上跑。

---

## CLI 参考

| 命令 | 说明 |
|------|------|
| `moe-l2 start --model <路径> --l2-size <大小>` | 启动代理 + 缓存 |
| `moe-l2 start --model <路径> --gpu` | 使用 GPU 加速的 llama-server 启动 |
| `moe-l2 stats --port <端口>` | 查看实时缓存统计 |
| `moe-l2 download-bins [--release TAG]` | 从 GitHub 下载预编译 GPU 二进制 |
| `moe-l2 collect --model <路径>` | 采集 MoE 路由数据 → `~/.moe-l2/maps/domain_expert_map.json` |
| `moe-l2 stop --port <端口>` | 停止代理 |

可选参数：
- `--model auto`：自动扫描 `/opt/data/models/*.gguf`
- `--l2-size 4GB` / `--l2-size 512MB`：目标缓存大小
- `--port 11435`（默认）
- `--gpu`：启用 GPU 模式（需要 CUDA + NVIDIA 显卡）

> **GPU 二进制**：不在 git 中追踪（Linux 为 `llama_bins.tar.gz`、Windows 为 `llama_bins_win.zip`，bins-v0.8.0 起多架构 sm_60/61/70/75/86/89/120a 一个包兼容全部 NVIDIA 卡含 P100/V100，含 cuda-libs / CUDA runtime DLL），运行时通过 `moe-l2 download-bins` 获取（PyPI 0.12.0 起按平台自动选择）。

---

## 平台要求

- **Linux x86_64 + NVIDIA 显卡** — 主平台（CUDA `.so` + `llama-server` 二进制包）
- **Windows 10/11 x64 + NVIDIA 显卡** — bins-v0.8.0 起原生支持（`llama_bins_win.zip`：llama-server.exe + DLL 全家，与 Linux 同款 A3 专家缓存 / 路由表功能）
- macOS、ARM Linux **暂不支持**
- **强烈建议使用 NVMe 固态硬盘**
- `--gpu` 模式需要 NVIDIA 显卡（CUDA 后端）

### Windows 原生版（bins-v0.8.0+）

正常安装 moe-l2 后拉取 Windows 引擎——`download-bins` 在 Windows 上会自动下载 `llama_bins_win.zip`：

```
pip install moe-l2
moe-l2 download-bins
moe-l2 start --model C:\models\Qwen3.6-35B-A3B-UD-IQ2_M.gguf --gpu
```

zip 内含 `llama-server.exe` + `llama-cli.exe` + 全部 DLL（多架构 cubin：sm_60/61/70/75/86/89/120a，AVX2，已含 CUDA 12.8 runtime DLL——无需安装 CUDA Toolkit，NVIDIA 驱动 ≥ 570 即可）。Windows 上生成领域路由表：`moe-l2 collect --model <你的.gguf>`（自动使用捆绑的 llama-cli.exe）。已在 RTX 3060 12G 实测（Qwen3.6-35B-A3B LRU 热缓存 ~21-22 t/s，与 Linux 同款引擎）。

---

## 更多数据

| 指标 | 标准 | moe-l2 |
|------|------|--------|
| Prompt 处理（DS-V2-Lite） | 110 t/s | 99 t/s · **308 t/s**（sched-cache=0.25） |
| 生成速度（DS-V2-Lite） | 65 t/s | 139-154 t/s · 39.2 t/s（sched-cache=0.25，08-02） |
| 生成速度（Qwen3.6-A3B） | — | 25.5-44.2 t/s |
| 显存占用 | 23.3 GB | **~5.4-10.1 GB** |

速度取舍是可预期的：专家驻留 CPU RAM（mmap 惰性 + on-demand pin，零显存），调度器每步只把激活的专家拷到 GPU。2026-08-19 bins-v0.6.0 主路径：DS-V2-Lite 生成 **139-154 t/s**、Qwen3.6-A3B **25.5-44.2 t/s**（混合领域）；DS 开 sched-cache=0.25 后 prompt 处理 308 t/s（+211%，08-02 口径）。

---

## 相关工作

### AirLLM（lyogavin/airllm，~29k stars）

AirLLM 是通用型超大模型分层加载方案，以 **Transformer 整层**为调度粒度：推理全程仅在显存保留单一层权重，其余落盘交换，实现极致低显存门槛（4GB 跑 70B）。但存在三个短板：① 每生成一个 Token 都要反复读写磁盘加载/释放整层权重，IO 开销巨大，对话生成速度极低；② 无 MoE 专属路由预测与专家热缓存（2026-07 才开始逐专家流式加载，Kimi K3），重复提问持续触发大量磁盘读取；③ 基于 Hugging Face Transformers 原生开发，无内置 OpenAI 服务接口，难以直接对接 Open WebUI、LangChain 等工具链。

| 维度 | AirLLM | moe-l2 |
|------|--------|--------|
| 调度最小单元 | Transformer 完整网络层 | **MoE 独立专家（稀疏最优）** |
| 目标模型 | 全模型兼容（稠密 + MoE） | **深度优化 MoE（DeepSeek/Qwen/Mixtral）** |
| 底层权重格式 | Hugging Face 原生权重 | **GGUF（llama.cpp 生态）** |
| 运行平台 | Windows/macOS/Linux 全兼容 | Linux x86_64 + NVIDIA 显卡 |
| MoE 场景内存 | 整层落盘交换，无热缓存 | **85GB V4：VRAM 8.3GB + RSS 11-12GB 封顶（实测）** |
| MoE 场景速度 | 逐层反复磁盘交换，适合批量离线 | 热专家缓存减少磁盘 IO，支持实时对话（Qwen 全链路 9.3 t/s 实测） |
| 服务接口 | 仅 Python 代码调用，无内置 Web 服务 | **内置 OpenAI 兼容代理（11435），开箱即用** |
| 显卡适配 | 原生 transformers，CUDA 适配繁琐 | **download-bins 多架构内核，10 系~50 系 N 卡全覆盖** |
| 超大分片模型 | 无针对性适配 | **原生修复多分片元数据解析 BUG，85GB 3 分片 V4 稳定** |

**选择建议**：选 moe-l2——本地跑 DeepSeek/Qwen 等 MoE 日常聊天、8G~12G 老消费 N 卡兼顾显存与速度、需要 OpenAI API 对接工具链、使用多分片超大 GGUF。选 AirLLM——需要运行稠密大模型、使用 Windows/macOS/AMD 或 CPU-only 环境（moe-l2 当前仅支持 Linux + NVIDIA）、仅一次性批量生成、只能用原生 HF 权重。

---

## 测试

每次 push 自动跑 CI（GitHub Actions，Python 3.10–3.13）：ruff 静态检查 + pytest 覆盖率（低于 50% 判失败）+ 打包验证。状态徽章：[![CI](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml/badge.svg)](https://github.com/yalun753/moe-l2/actions/workflows/ci.yml)

- **113 个测试**覆盖 Python 调度核心：领域预测器（关键词边界、兜底）、L2 缓存（LRU 淘汰、pin、领域切换）、GGUF 权重读取（合成模型）、透明代理（真实假后端 HTTP，阻塞 + SSE 流式）、CLI 辅助函数与数据飞轮。
- **覆盖率**：核心模块 72–88%（cache 88%、proxy 78%、gguf_reader 73%、predictor 72%），总计约 55%。
- 本地运行：
  ```bash
  uv sync --group dev
  uv run pytest tests/
  uv run ruff check moe_l2/ tests/
  ```

> C++ 侧（llama.cpp on-demand-pin / expert-cache 补丁）依赖 GPU，由 `references/` 下的端到端实测报告验证——见 [models-benchmark.md](references/zh/models-benchmark.md)。

## 项目状态

- ✅ 领域预测器（关键词 + 可选语义）
- ✅ L2 缓存（mmap LRU、线程安全、异步预加载）
- ✅ 透明代理（HTTP/SSE 转发）
- ✅ CLI（start/stats/collect/embed-map/download-bins，自动模型检测，GPU 模式）
- ✅ **selective pin + GPU 预填充（2026-08-10，v0.4.0，当前主路径）**：路由表驱动 top-K pin → V4 RSS **84.4 → 26.8GB**（on-demand 兜底 17.5GB；⚠️ V4 速度 N/A — 上游 deepseek4 CUDA 专家 bug #25582）；DS **139-154** / Qwen **25.5-44.2** t/s（4090，2026-08-19 bins-v0.6.0）；GPU cache 预填充让冷启动 round1 10.7 → 19.7 t/s（+84%）。历史里程碑：host-buffer 直算（08-02）→ on-demand pin（08-07）→ selective pin（08-10）
- ✅ **bins-v0.6.0（2026-08-19）**：cache 槽上限 16384→32768；per-slot 锁优化（并发正确性恢复——v0.5.0 无锁版 DS 并发必垃圾，已修复；同机 A/B 4090：Qwen 31.76 vs 32.42 持平、**DS 140.43 vs 113.15 +24%**）；保留热专家 v2（默认单表，`MOE_L2_POOL_SIZE=3` 可临时开启）；显存自适应主表 top-k（≤12G=75 / >12G=100）；proxy 并发修复（DS long 500 消失）。三卡回归（2080Ti/4090/5090）输出全正常。
- ✅ cache 挂 sched 拷贝层（2026-08-02）：DS 类模型 Prompt 99 → 308 t/s（+211%，cache=0.25，VRAM 不变）；Qwen/Mixtral 无收益不开
- ✅ **DeepSeek-V4-Flash（157B MoE）验证通过（2026-08-05）**：85GB 三片 GGUF 在 2080 Ti（11GB）上跑通——VRAM 8.3-9.1GB、RSS 靠专家页淘汰 v3.1 封顶（`MOE_L2_LRU_MAX_EXPERTS` 固定专家数 LRU）、多分片 GGUF 解析修复已随 0.7.0 发布。[完整报告](references/zh/deepseek-v4-flash-verify-20260805.md)
- ✅ PyPI 包（`moe-l2`）

---

## 许可证

**Apache 2.0。** 详见 [LICENSE](LICENSE)。

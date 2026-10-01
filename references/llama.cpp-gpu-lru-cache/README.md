# moe-l2 C++ 补丁源码（llama.cpp 侧）

这些是 moe-l2 对 llama.cpp 的**补丁源码**（不是完整树），照下表放回源树对应路径后再编译。

## 一、源树血缘（必须对齐，否则编出来的引擎和发布包不是一回事）

- 源树 = **qwen4exp 树，ggml 0.22.0**（2026-08-28），即 bins-v0.7.0 / v0.8.0
  Linux(`llama_bins.tar.gz`) 与 Windows(`llama_bins_win.zip`) 两个发布包的引擎树
- 发布包文件名自带版本：`libggml-base.so.0.22.0` / `libggml-base.dll`
- 原始（未打补丁）文件 md5，可用于校验手里的树是否同源：

| 树内路径 | md5 |
|---|---|
| `ggml/src/ggml-backend.cpp` | `1838200d080837b7ed98df8855850413` |
| `tools/server/server.cpp` | `ba479119ff89cea08eea2f4a1df69278` |
| `ggml/src/ggml-cuda/expert-cache.cu` | 见本目录同名文件（未被本补丁改动） |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 见本目录同名文件（未被本补丁改动） |

⚠️ 历史坑：2026-10-01 之前本目录存的是**另一条 0.17 线**的副本
（`ggml-backend.cpp` 里 `in_router = !g_router_mode`；0.22 线是 `const bool in_router = true;`），
照那份重建会得到与发布包不同的引擎。老副本已留档在 `bak-20261001-pre022/`。

## 二、文件落位

| 本目录文件 | 源树路径 |
|---|---|
| `ggml-backend.cpp` | `ggml/src/ggml-backend.cpp` |
| `server.cpp` | `tools/server/server.cpp` |
| `expert-cache.cu` / `expert-cache.cuh` | `ggml/src/ggml-cuda/` |
| `ggml-cuda.cu` | `ggml/src/ggml-cuda/ggml-cuda.cu` |

## 三、构建（只换两个库时的最小口径）

`ggml-backend.cpp` 编进 **libggml-base**、`server.cpp` 编进 **libllama-server-impl**；
要替换发布包里的这两个库时，只需编 `ggml-base` + `llama-server-impl` 两个目标，
**不必重编 CUDA 架构**（这两个文件里 `GGML_USE_CUDA` 出现 0 次，ABI 与带 CUDA 编译一致；
CUDA 后端由 `ggml_backend_load_all()` 在运行时加载现成的 `libggml-cuda.so`）。

```bash
cmake -S <tree> -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CUDA=OFF \
  -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
cmake --build build --target ggml-base llama-server-impl -j4
```

**glibc 必须对齐目标机**：在 gcc14/glibc2.41 上直接编会产出要求 `GLIBC_2.38`（`__isoc23_strtol` 族）
的库，跑在 glibc 2.35 的机器上会 `undefined symbol`。用与目标同代的容器编：
`docker run --rm -v <HOSTDIR>:/w ubuntu:22.04 bash /w/in_container_build.sh`
（自检：`objdump -T <lib> | grep -o "GLIBC_[0-9.]*" | sort -V -u | tail -1` 必须 ≤ 目标 glibc）。

## 四、已验证产物（进入 bins-v0.8.1 的两个库，2026-10-01）

| 库 | md5 | 说明 |
|---|---|---|
| `libggml-base.so.0.22.0` | `f79a07e4673bcc7addc6739f90be289b` | 含原子换表补丁 |
| `libllama-server-impl.so` | `7fc9211d81ee5c091d1c4a079de858cf` | 含 `commit()` 接线 |

实测（AutoDL 3080Ti 12G）：40 次生成中途换表全部成功、引擎日志 40 行 `router map committed`；
对照未打补丁的原版库**第 2 次换表即把 llama-server 打死**。

## 五、当前补丁内容（2026-10-01 起）

1. **原子换表**（`ggml-backend.cpp` + `server.cpp`）：全局 `g_router_map` 改为
   "写侧构建稿 + 读侧 `shared_ptr` 快照"，新增 `ggml_backend_router_commit()`
   （`atomic_store` 一次性发布），三个读点（pin / prefill / miss 回写）持快照再查，
   `g_prefill_done` 的 check-then-insert 加锁；`server.cpp` 在 `/moe-set-domain`
   全部 `set_layer()` 之后调 `commit()`。
   旧实现 `clear()` 先清空全局表再逐层回填，与推理线程并发读 → 越界
   （Windows 实测 `ggml-base.dll 0xc0000005 @0x175ae`；Linux 实测换表两次即崩）。
2. 早前已有（未变）：selective-pin / router map 加载、per-slot 锁、soft_resize、
   prefill 查重跳过、`/moe-set-domain` 的 retain 支持。

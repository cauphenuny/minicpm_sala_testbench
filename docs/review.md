# MiniCPM-SALA Commit 学习指南

> 本文档基于 sglang 仓库最近的 3 个 commit,梳理 MiniCPM-SALA(Sparse Attention + Linear Attention)特性的功能范围、修改文件,并提供由浅入深的阅读顺序,方便新手入手。

---

## 一、这些 commit 做了什么

一句话:**让 SGLang 支持 MiniCPM-SALA 模型** —— 一种 **混合注意力** 架构(部分层用 Sparse Attention 稀疏注意力,部分层用 Linear/GLA 线性注意力),并支持 CUDA Graph、工具调用(XML 解析)等。

涉及的 3 个 commit(按依赖顺序):

| Commit | 标题 | 说明 |
|--------|------|------|
| `8776894` | feat(jit_kernel): add minicpm_sala get_block_table kernel | JIT 编译的 CUDA kernel(计算 block table) |
| `ad80e7e` | feat(minicpm-sala): add MiniCPM sparse attention support | **核心功能**,~8000 行,稀疏注意力 + 内存池 + 运行时集成 + 工具调用 |
| `116d05e` | chore(minicpm-sala): add build docs and helper scripts | Dockerfile、安装脚本、中文文档 |

依赖关系:`8776894`(JIT kernel) → `ad80e7e`(主功能,调用 JIT kernel) → `116d05e`(文档)。

---

## 二、修改的文件清单

### Commit `116d05e` — 文档与构建
| 文件 | 作用 |
|------|------|
| `README_zh.md` | 中文文档(安装、启动参数、Q&A) |
| `Dockerfile.minicpm_sala` | Docker 镜像构建 |
| `install_minicpm_sala.sh` | 一键安装脚本 |

### Commit `ad80e7e` — 主功能(~8000 行)

**配置 / 接口层**
| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/srt/configs/__init__.py` | +2 | 导出配置类 |
| `python/sglang/srt/configs/mamba_utils.py` | +36 | SimpleGLA cache 参数 |
| `python/sglang/srt/configs/minicpm.py` | +199 | `MiniCPMHybridConfig` 模型配置类 |
| `python/sglang/srt/server_args.py` | +25 | 新增命令行参数 |
| `python/sglang/srt/utils/hf_transformers/common.py` | +2 | HF config 适配 |

**模型结构**
| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/srt/models/minicpm.py` | +296 | `MiniCPMForCausalLM` 模型定义 |

**注意力后端(核心)**
| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/srt/layers/attention/attention_registry.py` | +23 | 后端注册入口 |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | +247 | Linear/GLA 层后端 |
| `python/sglang/srt/layers/attention/minicpm/__init__.py` | 0 | 包标识 |
| `python/sglang/srt/layers/attention/minicpm/backend.py` | +2207 | **Sparse Attention 主后端** |
| `python/sglang/srt/layers/attention/minicpm/sparse_utils.py` | +1717 | 稀疏元数据计算 |
| `python/sglang/srt/layers/attention/minicpm/sparse_kernels.py` | +789 | Sparse kernel Python 封装 |
| `python/sglang/srt/layers/attention/minicpm/fuse_kernel.py` | +797 | 融合 kernel |
| `python/sglang/srt/layers/attention/minicpm/attention_kernels.py` | +522 | Attention kernel |

**内存池 / KV cache**
| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/srt/mem_cache/memory_pool.py` | +328 | Sparse KV cache 内存池 |
| `python/sglang/srt/mem_cache/common.py` | +44 | KV cache 公共逻辑 |
| `python/sglang/srt/mem_cache/chunk_cache.py` | +6 | Chunk cache 适配 |
| `python/sglang/srt/mem_cache/kv_cache_builder.py` | +1 | KV cache builder |
| `python/sglang/srt/mem_cache/mamba_radix_cache.py` | +25 | Mamba radix cache 适配 |

**运行时集成**
| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/srt/model_executor/model_runner.py` | +9 | ModelRunner 入口 |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | +57 | Sparse KV cache 初始化 |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | +2 | CUDA Graph replay 支持 |

**工具调用**
| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/srt/function_call/function_call_parser.py` | +2 | 解析器注册 |
| `python/sglang/srt/function_call/minicpm4_xml_detector.py` | +320 | MiniCPM XML 工具调用解析 |

**测试**
| 文件 | 行数 | 作用 |
|------|------|------|
| `test/registered/unit/layers/test_minicpm_sparse_metadata.py` | +92 | 稀疏元数据单测 |
| `test/registered/function_call/test_minicpm4_xml_detector.py` | +265 | 工具调用解析单测 |

### Commit `8776894` — JIT Kernel
| 文件 | 行数 | 作用 |
|------|------|------|
| `python/sglang/jit_kernel/csrc/minicpm_sala/get_block_table.cuh` | +311 | CUDA 源码 |
| `python/sglang/jit_kernel/minicpm_sala/__init__.py` | +11 | 包入口 |
| `python/sglang/jit_kernel/minicpm_sala/get_block_table.py` | +102 | Python 封装 |
| `python/sglang/jit_kernel/utils.py` | +30 | 工具函数 |
| `test/sglang/jit/benchmark/bench_get_block_table.py` | +102 | benchmark |
| `test/registered/jit/test_get_block_table.py` | +132 | 单测 |

---

## 三、推荐阅读顺序(由浅入深、由外到内)

### 第 0 层:先跑起来(理解"是什么")

| 顺序 | 文件 | 作用 |
|------|------|------|
| 1 | `README_zh.md` | **从这里开始**。中文文档,讲清了怎么安装、启动参数含义、整体目录结构 |
| 2 | `Dockerfile.minicpm_sala` / `install_minicpm_sala.sh` | 环境怎么搭的(可粗看) |

### 第 1 层:用户接口(理解"怎么用")

| 顺序 | 文件 | 作用 |
|------|------|------|
| 3 | `python/sglang/srt/server_args.py`(新增 25 行) | 新增的命令行参数,如 `--minicpm-dense-as-sparse`、`--attention-backend minicpm_flashinfer` 等 |
| 4 | `python/sglang/srt/configs/minicpm.py` | **模型配置类** `MiniCPMHybridConfig`。看它的 `__init__` 字段就能了解模型有哪些超参(`sparse_topk`、`sparse_window_size`、`lightning_nh` 等),是理解模型结构的入口 |

### 第 2 层:模型结构(理解"算什么")

| 顺序 | 文件 | 作用 |
|------|------|------|
| 5 | `python/sglang/srt/models/minicpm.py` | **模型定义**。看 `MiniCPMForCausalLM` 如何把 embedding + 多层(decoder layers)+ lm_head 拼起来,以及每层如何根据 `mixer_types` 选择 sparse 还是 linear 注意力 |

### 第 3 层:注意力后端(核心,理解"怎么算")

这是改动量最大的部分。建议按调用链顺序看:

| 顺序 | 文件 | 行数 | 作用 |
|------|------|------|------|
| 6 | `python/sglang/srt/layers/attention/attention_registry.py` | +23 | 后端注册入口,看 `minicpm_flashinfer` 怎么被注册的 |
| 7 | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | +247 | Linear/GLA 层的后端(相对简单,先看这个建立概念) |
| 8 | `python/sglang/srt/layers/attention/minicpm/backend.py` | **2207** | **Sparse Attention 主后端**,最核心最复杂。建议先读类/方法签名和注释,再深入 forward |
| 9 | `python/sglang/srt/layers/attention/minicpm/sparse_utils.py` | +1717 | 稀疏元数据计算:决定"每个 token 该看哪些 KV block" |
| 10 | `.../minicpm/sparse_kernels.py` / `fuse_kernel.py` / `attention_kernels.py` | 789/797/522 | CUDA kernel 的 Python 封装,可按需深入 |

### 第 4 层:内存与运行时集成(理解"怎么和 SGLang 框架对接")

| 顺序 | 文件 | 作用 |
|------|------|------|
| 11 | `python/sglang/srt/mem_cache/memory_pool.py`(+328) | Sparse KV cache 的内存池实现 |
| 12 | `python/sglang/srt/mem_cache/common.py` / `chunk_cache.py` / `mamba_radix_cache.py` | KV cache 构建与 radix 缓存的适配 |
| 13 | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`(+57) | ModelRunner 如何初始化 sparse KV cache |
| 14 | `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | Decode 阶段 CUDA Graph replay 支持 |

### 第 5 层:JIT Kernel(commit `8776894`)

| 顺序 | 文件 | 作用 |
|------|------|------|
| 15 | `python/sglang/jit_kernel/minicpm_sala/get_block_table.py` + `get_block_table.cuh` | 运行时按需编译的 CUDA kernel(计算 block table) |

### 第 6 层:工具调用 & 测试(理解"附加功能 + 如何验证")

| 顺序 | 文件 | 作用 |
|------|------|------|
| 16 | `python/sglang/srt/function_call/minicpm4_xml_detector.py` | MiniCPM XML 格式的工具调用解析 |
| 17 | `test/registered/unit/layers/test_minicpm_sparse_metadata.py` | 单测:稀疏元数据 |
| 18 | `test/registered/function_call/test_minicpm4_xml_detector.py` | 单测:工具调用解析 |
| 19 | `test/registered/jit/test_get_block_table.py` | 单测:JIT kernel |

---

## 四、核心调用链路(主线)

理解一次 forward 的核心路径:

```
models/minicpm.py            (模型定义,选择 sparse / linear 层)
    ↓
attention_registry.py        (后端注册与分发)
    ↓
minicpm/backend.py           (Sparse Attention 主后端 forward)
    ↓
minicpm/sparse_utils.py      (计算稀疏元数据:每个 token 看哪些 KV block)
    ↓
minicpm/{sparse_kernels, fuse_kernel, attention_kernels}.py  (CUDA kernel 执行)
```

辅助路径:
- **内存**:`memory_pool.py` → `model_runner_kv_cache_mixin.py`(初始化 sparse KV cache)
- **CUDA Graph**:`decode_cuda_graph_runner.py`(decode 阶段图捕获与重放)
- **JIT**:`jit_kernel/minicpm_sala/get_block_table.py`(运行时编译 block table kernel)

---

## 五、实操建议

1. **先把 README 跑通** —— 有了能运行的 server,看代码才有底。
2. **主线是注意力计算链路**:`models/minicpm.py` → `attention_registry.py` → `minicpm/backend.py` → `sparse_utils.py` → kernels。这条链路串起来就是一次 forward 的核心。
3. **`backend.py`(2207 行)是难点** —— 不要一上来通读。建议:
   - 先看类的 docstring 和 `forward` 方法主流程
   - 再逐个展开辅助方法
   - 可结合 `test_minicpm_sparse_metadata.py` 反推各字段含义
4. **三个 commit 的依赖关系**:`8776894`(JIT kernel) → `ad80e7e`(主功能,调用 JIT kernel) → `116d05e`(文档,无代码依赖)。所以 **功能上按前两个 commit 的顺序看** 即可。
5. **结合测试理解代码**:`test_minicpm_sparse_metadata.py`(92 行)和 `test_minicpm4_xml_detector.py`(265 行)是快速理解功能的捷径。

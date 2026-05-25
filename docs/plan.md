# MiniCPM Sparse Attention 上游化：双 PR 分拆方案

## Context

OpenBMB/sglang fork 上的 MiniCPM Sparse + Linear Attention 工作分支当前 `HEAD = d29fb130c`，与 upstream `v0.5.12` 的 merge-base 为 `7dd679cbb`（2026-01-08，63 commits 前），累计净增约 8000 行（不含 `sparse_kernel` submodule）。本方案将这套改动拆成两个先后顺序的 PR 推回 upstream，由两位负责人分别承担：

- **PR 1（owner A）**：算子 + Attention Backend 层 —— kernel、编排、backend 注册。
- **PR 2（owner B）**：框架集成层 —— 模型定义、memory pool、scheduler / cuda graph hooks、CLI flag、function-call parser、构建脚本。

拆分依据：

1. PR 1 几乎全是新文件、新路径，唯一接入点是 `attention_registry.py` (+25 行)，对 upstream 现有文件几乎无修改。PR 2 触及 `scheduler.py`、`memory_pool.py`、`cuda_graph_runner.py` 等高变更频率文件，必须在 rebase 时与 upstream 抽象对齐。
2. PR 2 依赖 PR 1 已合并的 backend 类与注册名才能跑通端到端，先 1 后 2 是硬性顺序。
3. 算子层有边界明确的瑕疵（硬编码常量、重复 kernel 变体、dead 分支、缺单测），上游化前清理范围清晰；框架层存在 `models/minicpm.py:474` 的 prefill TopK TODO、memory_pool 中 MiniCPM 专属 K1/K2 buffer 等需要在上游化前抽象。

---

## PR 1 — Sparse Attention Kernel + Backend

**目标**：把 MiniCPM 的稀疏注意力计算栈作为 self-contained 模块上游化。Upstream 看到的形态是「新增一类可注册的 attention backend + 其依赖的 triton kernels + 两个 vendored CUDA 子模块」。**不**引入 MiniCPM 模型本身或 scheduler / memory 改动。

### 范围

**新增（submodule）**
- `3rdparty/sparse_kernel` → `https://github.com/OpenBMB/sparse_kernel.git` @ `d7c367eb`
- `3rdparty/infllmv2_cuda_impl` → 分支 `minicpm_sala`
- `.gitmodules` 对应条目

**新增（纯 triton kernels，~1.9k 行）**
- `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py` (713)
- `python/sglang/srt/layers/attention/minicpm_fuse_kernel.py` (628)
- `python/sglang/srt/layers/attention/minicpm_attention_kernels.py` (522)

**新增（backend 编排，~3.9k 行）**
- `python/sglang/srt/layers/attention/minicpm_sparse_utils.py` (1665)
- `python/sglang/srt/layers/attention/minicpm_backend.py` (1950)
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` (262)

**修改（接入面，~25 行）**
- `python/sglang/srt/layers/attention/attention_registry.py`

**新增（测试，需补写）**
- `test/srt/attention/test_minicpm_sparse_kernels.py`
- `test/srt/attention/test_minicpm_backend.py`

### 执行步骤

#### 阶段 A — 分支与代码导入

A1. `git fetch upstream` 并确认本地 `v0.5.12` tag 与远端一致。

A2. 从 `v0.5.12` 创建新分支：`git checkout -b minicpm-sparse-backend-upstream v0.5.12`。

A3. 添加 `sparse_kernel` submodule：`git submodule add https://github.com/OpenBMB/sparse_kernel.git 3rdparty/sparse_kernel`，固定到 commit `d7c367eb`。

A4. 添加 `infllmv2_cuda_impl` submodule（分支 `minicpm_sala`），路径 `3rdparty/infllmv2_cuda_impl`。

A5. 把 `.gitmodules` 中两个 submodule 条目核对干净（无多余 path / fork-only 注释）。

A6. 从 fork 分支 `d29fb130c` 拷贝以下 6 个文件到新分支对应路径：
- `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`
- `python/sglang/srt/layers/attention/minicpm_fuse_kernel.py`
- `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`
- `python/sglang/srt/layers/attention/minicpm_sparse_utils.py`
- `python/sglang/srt/layers/attention/minicpm_backend.py`
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

A7. 把 `attention_registry.py` 中与 minicpm 相关的 +25 行注册条目移植过来，**只移植注册行**，不带任何 model 检测逻辑。

A8. 跑 `pip install -e .` + `python -c "from sglang.srt.layers.attention import minicpm_backend"` 确认 import 链通畅、submodule 已 build。

#### 阶段 B — 代码质量清理

B1. 删除 `minicpm_sparse_kernels.py:698-705` 的 dead `if True:` 分支及其下方未触发的 else。

B2. 删除 `minicpm_sparse_kernels.py:521-525` 的 `USE_TRITON_KERNEL` / `_COMPARISON_ENABLED` 环境变量 dev switch；如确需保留对照路径，挪到 `tests/` 下独立 helper。

B3. 评估 `compress_k_complete_kernel_new`（`:9-276`）与 padded 变体（`:280-545`）能否合并：
- 如能用 `tl.constexpr` 静态开关合并，提交合并 patch；
- 如不能（cudagraph 静态 shape 约束），在两份 kernel 上方加注释解释为何必须保留两份，并把共享逻辑抽到 helper Python 函数。

B4. 把 `minicpm_sparse_utils.py:150-155` 与 `:287-290` 的 `k1_stride=16, k1_l=32, k2_stride=64, k2_l=128` 提到 `SparseConfig` 类顶部常量；只保留一处定义，两处引用都改成读取常量。

B5. 给 `minicpm_sparse_utils.py:107` 的 `MAX_GRID_CHUNKS = 1024` 与 `:457` 的 `required_ratio = 16` 加注释说明取值依据（cudagraph buffer 上限 / 论文引用 / 经验值）。

B6. 清除 `minicpm_sparse_kernels.py:7`、`minicpm_sparse_utils.py:158` 的残留 TODO：要么完成，要么转成 GitHub issue 链接并在注释中引用。

B7. 给 `minicpm_sparse_utils.py` 中所有 Python helper（`batched_gather`、cache len 计算函数等）补全 `Tensor` 类型 hint。

B8. 跑 `ruff check python/sglang/srt/layers/attention/minicpm_*.py` 与项目既有 lint，修复所有报错。

#### 阶段 C — 补写单元测试

C1. 在 `test/srt/attention/test_minicpm_sparse_kernels.py` 中加入 reference torch 实现：
- `_ref_compress_k(k, stride, length)` 用 eager loop 实现 compress K
- `_ref_fuse_topk(scores, k)` 用 `torch.topk` 实现对照

C2. 测试用例：
- `test_compress_k_correctness`：随机 shape，对比 triton vs ref，`atol=1e-3, rtol=1e-2`
- `test_compress_k_padded_variant`：padded 输入 shape 下，对比 triton vs ref
- `test_fuse_topk_correctness`：对比 triton TopK 与 `torch.topk` 索引
- `test_fuse_topk_chunked`：chunked 输入 shape 下对比

C3. 在 `test/srt/attention/test_minicpm_backend.py` 中加入 backend 单测：
- 用 mock `ForwardBatch` + minimal `SparseConfig` 实例化 `MiniCPMAttnBackend`
- `test_forward_decode_shape`：检查 decode 路径 output shape / dtype
- `test_forward_extend_shape`：检查 extend 路径 output shape / dtype
- `test_cudagraph_capture_replay`：把 backend forward 放进 `torch.cuda.graph()` capture，replay 一次，确认结果一致且无 OOM

C4. 跑 `pytest test/srt/attention/test_minicpm_*.py -v` 全绿。

#### 阶段 D — Commit 整理

D1. 把当前分支所有改动 reset 到 `v0.5.12`，重新分阶段提交，目标是 3 个语义化 commits：

- Commit 1 `feat: add sparse_kernel submodules for MiniCPM attention`
  - `.gitmodules` 改动
  - `3rdparty/sparse_kernel` 与 `3rdparty/infllmv2_cuda_impl` 子模块条目

- Commit 2 `feat: add MiniCPM sparse attention triton kernels and backend`
  - 3 个 kernel 文件
  - 3 个 backend / utils 文件
  - `attention_registry.py` 注册行

- Commit 3 `test: add unit tests for MiniCPM sparse kernels and backend`
  - `test_minicpm_sparse_kernels.py`
  - `test_minicpm_backend.py`

D2. 每个 commit 单独 `git show` 检查，确认无残留 dev / debug 内容。

#### 阶段 E — 验证

E1. `git submodule update --init --recursive` 在干净 checkout 下成功。

E2. `pip install -e .` + 编译 sparse_kernel C++ 部分通过。

E3. 跑完整 `pytest test/srt -k "not slow"` 不回归。

E4. 新加的 `test/srt/attention/test_minicpm_*.py` 全绿。

E5. 跑 `python scripts/check_format.py`（或 upstream 当前用的格式检查脚本）通过。

#### 阶段 F — 开 PR

F1. Push 到 fork 上的 `minicpm-sparse-backend-upstream` 分支。

F2. 在 upstream 仓库 compare 视图开 PR。标题：`[Feature] Add MiniCPM sparse attention backend (kernels + dispatch)`。

F3. PR body 包含：
- 简介：这是什么算子（compress K + fused TopK + sparse attention），来源（MiniCPM 模型）
- 范围说明：本 PR **只**含计算栈与 backend 注册，模型 / 调度集成会有后续 PR
- 依赖说明：vendored 两个 CUDA submodule 的原因
- Benchmark：在某个 dummy workload 上 vs eager attention 的速度对比（可选但推荐）
- Test plan：列出 C4、E3、E4 的执行结果

F4. 把 review 反馈逐条记录在 PR 评论里，分批 force-push 修复（保持 3 个 commits 的结构）。

---

## PR 2 — MiniCPM Model + Framework Integration

**目标**：在 PR 1 提供的 backend 基础上，加入 MiniCPM 模型定义、配置、memory pool、scheduler / cuda graph hooks、CLI flag、function-call parser，使 `--model-path minicpm-...` 端到端可跑。

### 范围

**模型与配置（~470 行新增）**
- `python/sglang/srt/models/minicpm.py` (+298, -25)
- `python/sglang/srt/configs/minicpm.py` (+166 新文件)
- `python/sglang/srt/configs/__init__.py` (+2)
- `python/sglang/srt/configs/mamba_utils.py` (+35)
- `python/sglang/srt/utils/hf_transformers_utils.py` (+2)

**Memory / KV cache（~430 行）**
- `python/sglang/srt/mem_cache/memory_pool.py` (+207, -64)
- `python/sglang/srt/mem_cache/common.py` (+96)
- `python/sglang/srt/mem_cache/chunk_cache.py` (+35)
- `python/sglang/srt/mem_cache/mamba_radix_cache.py` (+42)

**调度 / runner / forward batch**
- `python/sglang/srt/managers/schedule_batch.py` (+220, -4)
- `python/sglang/srt/managers/scheduler.py` (+1)
- `python/sglang/srt/model_executor/forward_batch_info.py` (+73)
- `python/sglang/srt/model_executor/model_runner.py` (+9, -1)
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` (+40)
- `python/sglang/srt/model_executor/cuda_graph_runner.py` (+1)
- `python/sglang/srt/layers/attention/flashinfer_backend.py` (+1)
- `python/sglang/srt/layers/radix_attention.py` (+5)

**CLI / server**
- `python/sglang/srt/server_args.py` (+32)

**Function call（拆为 PR 2.5）**
- `python/sglang/srt/function_call/minicpm4_xml_detector.py` (+305 新)
- `python/sglang/srt/function_call/function_call_parser.py` (+2)
- `test/registered/function_call/test_minicpm4_xml_detector.py` (+254 新带测试)

**构建 / 文档**
- 转写成 `docs/models/minicpm_sala.md`，去掉 `Dockerfile.minicpm_sala` / `install_minicpm_sala.sh`

### 执行步骤

#### 阶段 A — 前置条件与分支

A1. 确认 PR 1 已合并进 upstream `main`；记录合并后的 upstream commit SHA 作为基线。

A2. `git fetch upstream` 把 PR 1 已合并的状态拉到本地。

A3. 从最新 upstream `main` 创建分支：`git checkout -b minicpm-framework-upstream upstream/main`。

A4. 确认 PR 1 的 backend 注册名（如 `minicpm_sparse` / `minicpm_hybrid`）在新分支中可 import。

#### 阶段 B — 抽象层修复（在拷贝模型 / pool 代码前完成）

B1. **解决 `models/minicpm.py:474` 的 TODO**：实现 prefill 模式下的完整 TopK 计算，或确认该路径在 production 不会触发后明确移除。完成后该 TODO 注释必须从代码中消失。

B2. **抽象 memory_pool.py 中 MiniCPM 专属 buffer**：
- 阅读 upstream `HybridReqToTokenPool` 当前实现，了解其 `__init__` 签名与 `alloc()` 语义
- 把 fork 中 `MiniCPMHybridReqToTokenPool` 的 sparse K1/K2 buffer 分配（fork 文件 `:563-568`、`:616-628`）改写为继承时通过 `extra_buffers: Dict[str, BufferSpec]` 参数注入，而不是在子类里硬写 `self.k1 = torch.zeros(...)`
- 若 upstream 没有 `extra_buffers` 机制，先在 `HybridReqToTokenPool` 基类加该扩展点（作为本 PR 的一部分），再在子类用它

B3. **抽象 scheduler.py 的模型检测**：
- 阅读 upstream `scheduler.py` 当前的 hybrid SSM 检测条件
- 把 fork 中的 `or self.tp_worker.model_runner.minicpm_hybrid_config is not None`（约 fork 文件 `:619`）替换为复用现有 `cache_params` / hybrid layer 类型的统一检测，不新加模型名分支

B4. **评估能否吸收 flashinfer_backend 的 +1 行**：
- 检查 fork 中 `flashinfer_backend.py` 对 minicpm-hybrid 路径的特化具体在做什么
- 如果只是 begin_forward args 适配，看能否通过让 `MiniCPMAttnBackend`（PR 1 已合并）在自身内部处理掉，从而 PR 2 完全不动 flashinfer_backend.py
- 如果做不到，保留该 +1 行并在 PR 描述里解释

#### 阶段 C — 模型与配置导入

C1. 把 `python/sglang/srt/configs/minicpm.py` 完整拷贝过来（166 行新文件）。

C2. 把 `python/sglang/srt/configs/__init__.py` 中 MiniCPM 注册行加入。

C3. 把 `mamba_utils.py` 中 hybrid layer 类型解析的相关 +35 行加入。

C4. 把 `hf_transformers_utils.py` 中 config 自动识别的 +2 行加入。

C5. 拷贝 `python/sglang/srt/models/minicpm.py`（已应用 B1 的 TODO 修复版本）。

C6. 验证 `python -c "from sglang.srt.configs.minicpm import MiniCPMHybridConfig"` 通过。

#### 阶段 D — Memory / KV cache 导入

D1. 把 `mem_cache/common.py`、`chunk_cache.py`、`mamba_radix_cache.py` 的改动逐文件 cherry-pick；每个文件应用后单独 `git diff` 检查无残留 fork-only 内容。

D2. 把 `memory_pool.py` 改动（已应用 B2 抽象重构后的版本）逐 hunk 应用；解决与 upstream churn 的所有冲突。

D3. 在 unit test 层面：
- 跑 upstream 既有 `test/srt/mem_cache/test_*` 不回归
- 加 1 个测试：`test_minicpm_hybrid_pool_alloc`，验证 sparse buffer 分配 / 释放路径

#### 阶段 E — Scheduler / runner / forward batch 导入

E1. 把 `forward_batch_info.py` 的 +73 行 sparse 字段加入。

E2. 把 `model_runner.py` 与 `model_runner_kv_cache_mixin.py` 改动应用，注意 upstream 该路径可能已重构。

E3. 把 `cuda_graph_runner.py` 的 +1 行 forward_batch 透传应用；如 upstream 该函数签名已变化，相应调整。

E4. 把 `scheduler.py` 应用 B3 抽象后的版本。

E5. 把 `schedule_batch.py` 的 +220 行 sparse metadata 字段加入。

E6. 把 `radix_attention.py` 的 +5 行 sparse 标注加入。

E7. 把 `flashinfer_backend.py` 的 +1 行（如未被 B4 吸收）应用。

#### 阶段 F — CLI flag

F1. 把 `server_args.py` 的 +32 行加入，包括：
- `--force-dense-minicpm`
- `--split-stage1`
- `--dense-as-sparse`
- `SGLANG_FLASHINFER_WORKSPACE_SIZE` 环境变量在 help 中的说明

F2. 每个 flag 加 docstring，写明用途、默认值、与 minicpm 模型的关系。

#### 阶段 G — 文档转写

G1. 创建 `docs/models/minicpm_sala.md`，整合：
- fork README.md 中关于 minicpm_sala 的核心说明
- README_zh.md 的中文版本（作为同名 `_zh.md` 或在同一份里附中文段落，看 upstream 文档约定）
- 安装命令（pip + submodule init），**不**包含 fork-only install 脚本
- 启动示例：`python -m sglang.launch_server --model-path ... --attention-backend minicpm_sparse ...`

G2. 不要拷贝 `Dockerfile.minicpm_sala` 与 `install_minicpm_sala.sh`。

G3. 不要修改 upstream 根目录 `README.md`，除非 upstream 有 model list 表格需要加一行。

#### 阶段 H — 端到端验证

H1. `python -m sglang.launch_server --model-path <minicpm-sala-checkpoint> --attention-backend minicpm_sparse` 起服务成功。

H2. `curl localhost:30000/generate -d '...'` 跑 prefill + decode，输出文本与 HuggingFace reference 实现逐 token 对照（top-1 一致率应 > 95%）。

H3. 加 `--enable-cuda-graph` 重复 H2，输出一致。

H4. 加 `--chunked-prefill-size 2048` 用一个 8k 长度的 prompt 测试，与一次性 prefill 输出一致。

H5. 加 `--dense-as-sparse` flag，dense 层走 sparse 路径不报错且输出合理。

H6. 跑 upstream 既有 `pytest test/srt -k "not slow"` 全绿，不回归。

H7. 跑新加的 `test_minicpm_hybrid_pool_alloc` 等单测全绿。

#### 阶段 I — Commit 整理

I1. 重新分阶段提交，目标是 4 个语义化 commits：

- Commit 1 `feat: add MiniCPM hybrid model config and architecture`
  - `configs/minicpm.py`、`configs/__init__.py`、`mamba_utils.py`、`hf_transformers_utils.py`、`models/minicpm.py`

- Commit 2 `feat: add MiniCPM hybrid memory pool with sparse buffer support`
  - `memory_pool.py`、`mem_cache/common.py`、`chunk_cache.py`、`mamba_radix_cache.py`
  - 配套 pool 单测

- Commit 3 `feat: wire MiniCPM hybrid model into scheduler and runner`
  - `scheduler.py`、`schedule_batch.py`、`forward_batch_info.py`、`model_runner.py`、`model_runner_kv_cache_mixin.py`、`cuda_graph_runner.py`、`flashinfer_backend.py`、`radix_attention.py`

- Commit 4 `feat: add MiniCPM-specific CLI flags and docs`
  - `server_args.py`、`docs/models/minicpm_sala.md`

#### 阶段 J — 开 PR

J1. Push 到 fork 上的 `minicpm-framework-upstream` 分支。

J2. 在 upstream 仓库 compare 视图开 PR。标题：`[Feature] Integrate MiniCPM hybrid (sparse + linear) model into SGLang runtime`。

J3. PR body 包含：
- 简介：本 PR 在 PR 1 已合入的 backend 基础上接通模型 / 调度 / memory pool / CLI
- 依赖：链接到 PR 1 的合入 commit
- 抽象决策：B2 / B3 / B4 的设计说明（为什么这样抽象，是否对其他 hybrid 模型有 net positive）
- Test plan：H1–H7 的结果

J4. 在 PR 中明确说明 function-call parser 拆到 PR 2.5。

---

## PR 2.5 — Function Call XML Parser

**目标**：MiniCPM4 XML 工具调用解析器，与 attention 工作正交，独立提交。

### 范围

- `python/sglang/srt/function_call/minicpm4_xml_detector.py` (+305 新)
- `python/sglang/srt/function_call/function_call_parser.py` (+2)
- `test/registered/function_call/test_minicpm4_xml_detector.py` (+254 新)

### 执行步骤

P1. 从最新 upstream `main` 创建分支 `minicpm-xml-detector-upstream`。

P2. 拷贝三个文件到对应路径。

P3. 跑 `pytest test/registered/function_call/test_minicpm4_xml_detector.py -v` 全绿。

P4. 跑 upstream 既有 function call 测试不回归。

P5. 单 commit 提交：`feat: add MiniCPM4 XML function call detector`。

P6. 与 PR 2 可并行开。

---

## 全局执行顺序

```
[PR 1: kernel + backend] ────merge────┐
                                       ▼
              [PR 2: framework] ──┬─→ merge
              [PR 2.5: function call] ─┘（可与 PR 2 并行）
```

不要并行开 PR 1 / PR 2 —— PR 2 的 backend 引用、attention registry 注册名都来自 PR 1，先后顺序是硬约束。PR 2.5 与 PR 2 完全正交，可同时进行 review。

## 不要做的事

- ❌ 不要把 `install_minicpm_sala.sh` / `Dockerfile.minicpm_sala` 原样塞进 upstream PR
- ❌ 不要把 `_COMPARISON_ENABLED` / `USE_TRITON_KERNEL` 等 dev env switch 留在 PR 1
- ❌ 不要在 PR 1 内顺手提交 model file（`models/minicpm.py`），那是 PR 2 的内容
- ❌ 不要试图保留完整 63-commit 历史，按上述 squash 方案合并成语义化 commits
- ❌ 不要在 PR 2 里硬编码 `minicpm_hybrid_config is not None` 这类模型名检测，必须走抽象（B3）

### DEV 指南：TRELLIS 官方代码迁移与文件映射

本文档说明如何将 TRELLIS 官方代码内置到主仓库，并给出所有相关代码文件的映射关系与导入规范。

---

## 目标与约束
- **目标**：去除对 `_reference_codes/TRELLIS` 的动态路径依赖，统一走仓库内置包，导入路径尽量简短。
- **约束**：
  - 代码中避免 try/except 或任何 fallback。
  - 含张量运算的每行需附上“操作后张量形状”的注释。

---

## 目录架构（直接内置到 generators/trellis，短路径导入）
将上游 `trellis` 包中“实际用到的最小子集”直接放进 `generators/trellis/` 下的 `modules/` 与 `pipelines/`，并通过门面模块重导出常用入口，实现简短导入：

```
generators/
  trellis/
    __init__.py                 # 门面：重导出 sparse 与 TrellisImageTo3DPipeline
    pipeline.py                 # 本仓库的 Stage2 封装（调用上游 pipeline）
    utils.py                    # 预处理与 Mesh 转换等
    patches/
      sparse_tensor_utils.py    # 稀疏拼接等

    modules/
      __init__.py               # from . import sparse 作为命名空间导出
      sparse/
        __init__.py
        basic.py
        linear.py
        ops.py

    pipelines/
      __init__.py               # from .trellis_image_to_3d import TrellisImageTo3DPipeline
      trellis_image_to_3d.py
```

重写规则（当拷贝上游源码时）：
- 将源码里的绝对导入 `from trellis.xxx import ...` 改为“相对导入”，例如：
  - `from trellis.modules.sparse import basic` → `from ..modules.sparse import basic`
  - `from trellis.modules.sparse.linear import SparseLinear` → `from ..modules.sparse.linear import SparseLinear`
  - `from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline` → `from .trellis_image_to_3d import TrellisImageTo3DPipeline`

公开导入形式（推荐）：
- `from generators.trellis import sparse as sp`
- `from generators.trellis import TrellisImageTo3DPipeline`
- 若需具体类：`from generators.trellis.modules.sparse.linear import SparseLinear`

---

## 导入映射（旧 → 新）

| 旧导入 | 新导入（推荐） |
|---|---|
| `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp` |
| `from trellis.modules.sparse.linear import SparseLinear` | `from generators.trellis.modules.sparse.linear import SparseLinear` |
| `from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline` | `from generators.trellis import TrellisImageTo3DPipeline` |

说明：完成迁移后，需移除所有 `_reference_codes/TRELLIS` 的 `sys.path.insert(...)`。

---

## 文件映射总览（本仓库 ↔ 上游 TRELLIS）

### 公开入口与上游对应

| 本仓库文件 | 角色 | 上游对应 |
|---|---|---|
| `generators/trellis/pipeline.py` | 封装 Stage2 推理/训练入口 | `trellis/pipelines/trellis_image_to_3d.py`（构造/调用） |
| `generators/trellis/utils.py` | 图像预处理与 Mesh 转换工具 | 参考 `trellis/pipelines/trellis_image_to_3d.py` 预处理逻辑 |
| `generators/trellis/patches/sparse_tensor_utils.py` | 稀疏拼接/CFG 相关工具 | 基于 `trellis/modules/sparse/*` 的数据结构 |

### 内置代码与上游一一对应

| 本仓库（内置） | 上游官方 |
|---|---|
| `generators/trellis/modules/sparse/basic.py` | `_reference_codes/TRELLIS/trellis/modules/sparse/basic.py` |
| `generators/trellis/modules/sparse/linear.py` | `_reference_codes/TRELLIS/trellis/modules/sparse/linear.py` |
| `generators/trellis/modules/sparse/ops.py` | `_reference_codes/TRELLIS/trellis/modules/sparse/ops.py` |
| `generators/trellis/pipelines/trellis_image_to_3d.py` | `_reference_codes/TRELLIS/trellis/pipelines/trellis_image_to_3d.py` |

### 依赖 TRELLIS 的本仓库文件（需改为短路径导入）

| 本仓库文件 | 旧依赖（示例） | 新依赖（示例） |
|---|---|---|
| `generators/trellis/pipeline.py` | `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp`；`from generators.trellis import TrellisImageTo3DPipeline` |
| `generators/trellis/utils.py` | `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp` |
| `generators/trellis/patches/sparse_tensor_utils.py` | `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp` |
| `flow_grpo/diffusers_patch/trellis_stage2_with_logprob.py` | `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp` |
| `flow_grpo/diffusers_patch/sparse_tensor_grpo.py` | `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp` |
| `flow_grpo/diffusers_patch/trellis_flow_with_logprob.py` | `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp` |
| `flow_grpo/peft_sparse/sparse_lora.py` | `import trellis.modules.sparse as sp` / `from trellis.modules.sparse.linear import SparseLinear` | `from generators.trellis import sparse as sp` / `from generators.trellis.modules.sparse.linear import SparseLinear` |
| `scripts/test_trellis_suite.py` | `import trellis.modules.sparse as sp` | `from generators.trellis import sparse as sp` |

### 间接依赖（经封装调用 TRELLIS）

| 本仓库文件 | 依赖路径 |
|---|---|
| `scripts/train_trellis.py` | 通过 `generators.trellis.pipeline.TrellisStage2Pipeline` 与 `flow_grpo/diffusers_patch/*` 间接使用 TRELLIS |
| `config/trellis_stage2_grpo.py` | 绑定上游权重目录结构（如 `pretrained_weights/TRELLIS-image-large`） |

---

## 迁移步骤（建议）
1. 将上游最小子集拷贝至：
   - `trellis/modules/sparse/*` → `generators/trellis/modules/sparse/*`
   - `trellis/pipelines/trellis_image_to_3d.py` → `generators/trellis/pipelines/trellis_image_to_3d.py`
2. 在拷贝的源码中，将所有 `from trellis...` 绝对导入改为“相对导入”（见上文重写规则）。
3. 在 `generators/trellis/modules/__init__.py` 与 `generators/trellis/pipelines/__init__.py` 中重导出常用对象。
4. 在 `generators/trellis/__init__.py` 中统一二次导出（`sparse`、`TrellisImageTo3DPipeline`）。
5. 替换表格中列出的所有旧导入为新导入，并删除所有 `_reference_codes/TRELLIS` 的 `sys.path` 注入。
6. 跑 `scripts/test_trellis_suite.py` 做快速回归，再用少量步数跑 `scripts/train_trellis.py` 进行 smoke test。

---

## 运行与依赖提示
- 保持 `spconv-cu124`、`cumm` 等稀疏依赖；必要时在 `requirements.txt` 锁定版本。
- 如需本地/离线运行，上层封装可设置环境变量：`ATTN_BACKEND=xformers`、`HF_HUB_OFFLINE=1`。

---

## 备注
- 若后续扩展更多上游模块，直接增量添加到 `generators/trellis/modules/**` 或 `generators/trellis/pipelines/**`，并按需在门面层重导出即可。
- 与 Hunyuan3D 的做法一致：内置第三方实现 + 门面重导出，保证路径简洁与上游隔离。



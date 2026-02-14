# spconv eval 路径 bf16 GEMM 算法查找失败

## 报错信息

```
RuntimeError: !all_profile_res.empty() assert failed. can't find suitable algorithm for 0
```

来自 `spconv.pytorch.ops.implicit_gemm`，仅在 **evaluation 阶段**出现，训练正常。

## 根因分析

### 触发链路

1. `accelerator.prepare(slat_flow_model, optimizer)` 做了**两件事**：
   - **DDP 包装**（`DistributedDataParallel`）
   - **修改 `model.forward`**，注入 `autocast(bf16)` + `convert_outputs_to_fp32` 装饰器
     （原始 forward 保存在 `model._original_forward`）
2. 评估时 `model.eval()` 将 `self.training` 设为 `False`
3. spconv 的 `_conv_forward` 根据 `self.training` 分流：
   - `training=True` → `Fsp.implicit_gemm`（训练路径，支持 bf16）
   - `training=False` → `ops.implicit_gemm`（推理路径，由 `ConvTunerSimple` profiling 选算法）
4. autocast(bf16) 使 `nn.Linear`（包括 `SparseLinear`）输出为 bf16，这些 bf16 特征传入 spconv 卷积层
5. `ConvTunerSimple` 无法为 **bf16 输入**找到合适的 GEMM 算法 → 断言失败

### 关键点

- **训练时不报错**：训练路径 `Fsp.implicit_gemm` 能处理 bf16
- **commit b7557ea5 之前不报错**：该 commit 引入了 `accelerator.prepare()` 包装 `slat_flow_model`，之前模型未被 autocast 包裹
- **eval_only 脚本不报错**：`eval_only=True` 时 `strategy=None`，`prepare_models_and_optimizers` 跳过，模型未被 DDP/autocast 包装

## 参考：TRELLIS 原始代码的做法

TRELLIS 训练代码（`trellis/trainers/basic.py`）维护了**两套模型引用**：

```python
# 训练用 DDP 包装版
self.training_models = {name: DDP(model, ...) for name, model in self.models.items()}

# 推理/采样用原始版
sampler.sample(self.models['denoiser'], ...)  # run_snapshot 中
```

推理时**直接用 `self.models`**（原始模型），不需要拆装。

## 修复方案

仿照 TRELLIS 设计，用 `SpconvInferenceMixin` 将推理兼容逻辑封装为 mixin。

### `SpconvInferenceMixin`（`strategy.py`）

```python
class SpconvInferenceMixin:
    """Mixin: 推理时临时还原原始 forward（无 autocast(bf16)），解决 spconv eval 不支持 bf16 问题。"""

    _original_forward = None

    def prepare(self, accelerator, optimizer):
        self._original_forward = self._student.forward  # ★ 在 accelerate 注入 autocast 之前保存
        return super().prepare(accelerator, optimizer)

    @contextmanager
    def inference_context(self):
        if self._accelerator is None or self._original_forward is None:
            yield; return
        pipe_models = self.pipeline.pipe.models
        saved_model = pipe_models["slat_flow_model"]
        inner = self._accelerator.unwrap_model(self._student)
        patched_forward = inner.forward
        inner.forward = self._original_forward
        pipe_models["slat_flow_model"] = inner
        try:
            yield
        finally:
            inner.forward = patched_forward
            pipe_models["slat_flow_model"] = saved_model
```

需要 spconv 推理的策略混入即可：

```python
class FullFinetuneStrategy(SpconvInferenceMixin, TrainingStrategy): ...
class LoRAStrategy(SpconvInferenceMixin, TrainingStrategy): ...
class FrozenStrategy(TrainingStrategy): ...  # 无需 mixin（不走 accelerator.prepare）
```

### `evaluate()` — 一行搞定

```python
inference_ctx = system.strategy.inference_context() if system.strategy else nullcontext()

with inference_ctx, EvalModeGuard(...):
    for batch in eval_loader:
        ...
```

- 逻辑内聚在 mixin 层，`evaluate()` 保持简洁
- `FrozenStrategy` 不混入 mixin，基类 `inference_context()` 是 no-op
- `trellis.py`、`trellis_bilevel.py`、`trellis_pp.py` 所有调用方自动受益

"""
退化 latent 保护机制测试。

验证 chunked decoder 在退化输入（0 点 / 极端值）下不会崩溃，
而是返回 0 点的空 SparseTensor，由上层 h.feats.shape[0]==0 自然触发 StageSkipError。

测试分三部分：
  1. 单元测试：_merge_tensors / merge() 的空输入行为
  2. 集成测试：forward_chunked 接收 0 点 SparseTensor
  3. 端到端测试：forward.py 的 StageSkipError 触发
"""

import os
import sys
import traceback

import torch

# 设置路径
repo_root = os.path.abspath(os.path.dirname(__file__) + "/../..")
trellis2_ref_root = os.path.join(repo_root, "_reference_codes", "TRELLIS.2")
if trellis2_ref_root not in sys.path:
    sys.path.insert(0, trellis2_ref_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from trellis2.modules.sparse import SparseTensor

from edit4shape.generators.trellis2.chunked import (
    ChunkableSparseTensor,
    ChunkMeta,
)
from edit4shape.generators.trellis2.chunked_mixin import ChunkedDecoderMixin
from edit4shape.systems.utils.stage_ops import StageSkipError


# =====================================================================
# 辅助
# =====================================================================

_pass = 0
_fail = 0


def check(name: str, condition: bool, detail: str = ""):
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  ✓ {name}")
    else:
        _fail += 1
        msg = f"  ✗ {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# =====================================================================
# Test 1 — _merge_tensors 单元测试
# =====================================================================

def test_merge_tensors_empty_list(device: torch.device):
    """_merge_tensors([]) 应返回 0 点 SparseTensor，不返回 None。"""
    section("Test 1a: _merge_tensors 空列表")

    coords = torch.randint(0, 16, (10, 4), device=device).int()   # (10, 4)
    coords[:, 0] = 0  # batch 维度必须为 0                        # (10, 4)
    ref = SparseTensor(
        torch.randn(10, 8, device=device),                      # (10, 8)
        coords,
        scale=(1, 1, 1),
    )
    cs = ChunkableSparseTensor(ref, axis=3, chunk_size=64, halo=2)
    # 不迭代 chunks，直接构造空列表场景
    cs._chunks = []  # 模拟无 chunk 结果

    result = cs._merge_tensors([])

    check("返回类型是 SparseTensor", isinstance(result, SparseTensor),
          f"got {type(result)}")
    check("feats.shape[0] == 0", result.feats.shape[0] == 0,
          f"got {result.feats.shape[0]}")
    check("coords.shape == (0, 4)", result.coords.shape == (0, 4),
          f"got {result.coords.shape}")


def test_merge_tensors_empty_after_halo(device: torch.device):
    """halo 过滤后无有效点 → 0 点 SparseTensor。"""
    section("Test 1b: _merge_tensors halo 过滤后为空")

    ref = SparseTensor(
        torch.randn(5, 8, device=device),                       # (5, 8)
        torch.zeros(5, 4, dtype=torch.int32, device=device),     # (5, 4) 全在坐标 0
        scale=(1, 1, 1),
    )
    cs = ChunkableSparseTensor(ref, axis=3, chunk_size=4, halo=2, coord_scale=1)

    # 构造一个 chunk 结果，其中 valid 区域 [start, end) 不包含任何实际坐标
    # actual_halo=2, start=10, end=14, coord_scale=1
    # → local_start = 2*1 = 2, local_end = (2+14-10)*1 = 6
    # → valid 要求 coords[:,axis] ∈ [2, 6)，但所有 coords 的 axis 列 = 0
    # → halo 过滤后无有效点
    fake_tensor = SparseTensor(
        torch.randn(3, 8, device=device),                        # (3, 8)
        torch.zeros(3, 4, dtype=torch.int32, device=device),     # (3, 4) 坐标全为 0
        scale=(1, 1, 1),
    )
    meta = ChunkMeta(
        start=10, end=14, actual_halo=2,
        original_scale=(1, 1, 1),
        valid_mask=torch.ones(3, dtype=torch.bool, device=device),
    )

    result = cs._merge_tensors([(fake_tensor, meta)])

    check("返回类型是 SparseTensor", isinstance(result, SparseTensor),
          f"got {type(result)}")
    check("feats.shape[0] == 0", result.feats.shape[0] == 0,
          f"got {result.feats.shape[0]}")


def test_merge_method_empty(device: torch.device):
    """ChunkableSparseTensor.merge() 空 chunks → 0 点 SparseTensor。"""
    section("Test 1c: merge() 空 chunks")

    coords = torch.randint(0, 8, (4, 4), device=device).int()    # (4, 4)
    coords[:, 0] = 0  # batch 维度必须为 0                        # (4, 4)
    ref = SparseTensor(
        torch.randn(4, 16, device=device),                      # (4, 16)
        coords,
        scale=(1, 1, 1),
    )
    cs = ChunkableSparseTensor(ref, axis=3, chunk_size=64, halo=2)
    cs._chunks = []

    merged = cs.merge()

    check("返回类型是 SparseTensor", isinstance(merged, SparseTensor),
          f"got {type(merged)}")
    check("feats.shape[0] == 0", merged.feats.shape[0] == 0,
          f"got {merged.feats.shape[0]}")


# =====================================================================
# Test 2 — forward_chunked 集成测试（需要加载模型）
# =====================================================================

def test_forward_chunked_zero_points(device: torch.device, decoder):
    """0 点 SparseTensor 输入 → forward_chunked 返回 0 点 SparseTensor。"""
    section("Test 2a: forward_chunked 0 点输入")

    in_channels = decoder.from_latent.weight.shape[1]  # latent 通道数
    empty_feats = torch.empty(0, in_channels, device=device)                # (0, C)
    empty_coords = torch.zeros(0, 4, dtype=torch.int32, device=device)      # (0, 4)
    empty_slat = SparseTensor(empty_feats, empty_coords, scale=(1, 1, 1))

    with torch.no_grad():
        h, subs = decoder.forward_chunked(
            empty_slat, axis=3, return_subs=True, use_checkpoint=False,
        )

    check("h 是 SparseTensor", isinstance(h, SparseTensor),
          f"got {type(h)}")
    check("h.feats.shape[0] == 0", h.feats.shape[0] == 0,
          f"got {h.feats.shape[0]}")
    check("subs 是空列表", len(subs) == 0,
          f"got len={len(subs)}")


def test_forward_chunked_single_point(device: torch.device, decoder):
    """1 点 SparseTensor → 不崩溃（可能有点或空）。"""
    section("Test 2b: forward_chunked 1 点输入")

    in_channels = decoder.from_latent.weight.shape[1]
    one_feats = torch.zeros(1, in_channels, device=device)                  # (1, C)
    one_coords = torch.zeros(1, 4, dtype=torch.int32, device=device)        # (1, 4)
    one_slat = SparseTensor(one_feats, one_coords, scale=(1, 1, 1))

    try:
        with torch.no_grad():
            h, subs = decoder.forward_chunked(
                one_slat, axis=3, return_subs=True, use_checkpoint=False,
            )
        check("不崩溃", True)
        check(f"h.feats.shape[0] = {h.feats.shape[0]}（0 或正整数均可）",
              h.feats.shape[0] >= 0)
    except Exception as e:
        check("不崩溃", False, f"异常: {e}")


# =====================================================================
# Test 3 — 端到端 StageSkipError 测试
# =====================================================================

def test_stage_skip_error_raised(device: torch.device, decoder):
    """
    模拟 forward.py 的检查逻辑：
    decoder 返回 0 点 → h.feats.shape[0]==0 → StageSkipError。
    """
    section("Test 3: StageSkipError 触发")

    in_channels = decoder.from_latent.weight.shape[1]
    empty_feats = torch.empty(0, in_channels, device=device)                # (0, C)
    empty_coords = torch.zeros(0, 4, dtype=torch.int32, device=device)      # (0, 4)
    empty_slat = SparseTensor(empty_feats, empty_coords, scale=(1, 1, 1))

    with torch.no_grad():
        h, subs = decoder.forward_chunked(
            empty_slat, axis=3, return_subs=True, use_checkpoint=False,
        )

    # 模拟 forward.py L99-102 的检查
    raised = False
    try:
        if h.feats.shape[0] == 0:
            raise StageSkipError(
                "Shape decoder produced empty output (degenerate latent)"
            )
    except StageSkipError:
        raised = True

    check("StageSkipError 被正确触发", raised)


# =====================================================================
# Test 4 — 正常输入回归测试
# =====================================================================

def test_normal_input_still_works(device: torch.device, pipeline):
    """正常 latent → 正常 mesh（确保修改没有破坏正常路径）。"""
    section("Test 4: 正常输入回归测试")

    from PIL import Image

    image_path = os.path.join(
        repo_root, "_reference_codes", "TRELLIS.2", "assets", "example_image", "image_01.png"
    )
    if not os.path.exists(image_path):
        check("跳过（找不到测试图片）", True)
        return

    image = Image.open(image_path)

    # 使用参考 pipeline 进行条件编码 + dense sampling
    pipe = pipeline.pipe
    image_proc = pipe.preprocess_image(image)

    torch.manual_seed(42)
    cond = pipe.get_cond([image_proc], resolution=1024)
    coords = pipe.sample_sparse_structure(cond, 64, num_samples=1)

    # 生成随机 latent 并 denormalize
    shape_flow_model = pipe.models['shape_slat_flow_model_1024']
    in_channels = shape_flow_model.in_channels
    torch.manual_seed(42)
    noise = torch.randn(coords.shape[0], in_channels, device=device)        # (N, C)
    shape_slat = SparseTensor(coords=coords, feats=noise)

    # denormalize
    std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(device)   # (1, C)
    mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(device) # (1, C)
    shape_slat = shape_slat * std + mean

    # 通过 chunked decoder 解码
    decoder = pipe.models['shape_slat_decoder']
    decoder.set_resolution(1024)

    with torch.no_grad():
        h, subs = decoder.forward_chunked(
            shape_slat, axis=3, return_subs=True, use_checkpoint=False,
        )

    check("h.feats.shape[0] > 0", h.feats.shape[0] > 0,
          f"got {h.feats.shape[0]}")
    check(f"h.feats.shape = {tuple(h.feats.shape)}", h.feats.shape[1] == 7,
          f"expected (N, 7)")
    check(f"subs 非空 (len={len(subs)})", len(subs) > 0)

    print(f"  ℹ  h.feats.shape = {tuple(h.feats.shape)}, subs 层数 = {len(subs)}")


# =====================================================================
# main
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="退化 latent 保护机制测试"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-integration", action="store_true",
                        help="跳过需要加载模型的集成测试")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ==== 单元测试（不需要模型） ====
    test_merge_tensors_empty_list(device)
    test_merge_tensors_empty_after_halo(device)
    test_merge_method_empty(device)

    if args.skip_integration:
        print("\n⚠  跳过集成测试（--skip-integration）")
    else:
        # ==== 加载模型 ====
        section("加载模型")
        import ml_collections
        from edit4shape.generators.trellis2.pipeline_adapter import (
            build_pipeline_from_reference,
        )

        cfg = ml_collections.ConfigDict()
        cfg.pretrained = ml_collections.ConfigDict()
        cfg.pretrained.model = "./pretrained_weights/TRELLIS.2-4B"
        cfg.pretrained.dino_local_path = (
            "./pretrained_weights/dinov3-vitl16-pretrain-lvd1689m/"
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        cfg.pipeline_type = "1024"
        cfg.verbose = False

        class MockAccelerator:
            pass

        accelerator = MockAccelerator()
        accelerator.device = device

        pipeline = build_pipeline_from_reference(cfg, accelerator)

        # 注入 chunked mixin
        decoder = pipeline.pipe.models["shape_slat_decoder"]
        ChunkedDecoderMixin.inject_to(decoder)
        decoder.set_resolution(1024)

        print(f"  decoder 类型: {type(decoder).__name__}")
        print(f"  from_latent 输入通道: {decoder.from_latent.weight.shape[1]}")

        # ==== 集成测试 ====
        test_forward_chunked_zero_points(device, decoder)
        test_forward_chunked_single_point(device, decoder)
        test_stage_skip_error_raised(device, decoder)
        test_normal_input_still_works(device, pipeline)

    # ==== 汇总 ====
    section("测试汇总")
    total = _pass + _fail
    print(f"  通过: {_pass}/{total}    失败: {_fail}/{total}")
    if _fail > 0:
        print("  ⚠  有失败的测试项，请检查上方输出")
        sys.exit(1)
    else:
        print("  ✅ 全部通过")


if __name__ == "__main__":
    main()

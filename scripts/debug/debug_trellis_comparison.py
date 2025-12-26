#!/usr/bin/env python3
"""
Trellis Pipeline 单步对比调试脚本。

功能：对比 源代码 pipeline.run() 和 你的 adapter + rollout_sparse 在相同输入下的差异。

使用方法：
    cd /home/zhiyuan_ma/code/flow_grpo_custom
    python scripts/debug_trellis_comparison.py --image assets/example_image/T.png --seed 42
"""

import os
import sys
import argparse
from pathlib import Path

# 设置环境变量（在导入其他模块前）
os.environ['SPCONV_ALGO'] = 'native'

import torch
import numpy as np
from PIL import Image

# 添加项目根目录和 TRELLIS 参考代码路径
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRELLIS_REF_ROOT = REPO_ROOT / "_reference_codes" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_REF_ROOT))
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Trellis 对比调试")
    parser.add_argument("--image", type=str, default=str(TRELLIS_REF_ROOT / "assets/example_image/T.png"),
                        help="测试图片路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--model_path", type=str, default="./pretrained_weights/TRELLIS-image-large",
                        help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default="./debug_comparison_output",
                        help="输出目录")
    parser.add_argument("--mode", type=str, choices=["ref", "ours", "both"], default="both",
                        help="运行模式: ref=仅参考代码, ours=仅我们的代码, both=两者都运行")
    parser.add_argument("--ref_gpu", type=int, default=0, help="参考代码使用的 GPU ID")
    parser.add_argument("--our_gpu", type=int, default=1, help="我们代码使用的 GPU ID")
    return parser.parse_args()


def tensor_stats(t, name="tensor"):
    """打印 tensor 统计信息"""
    if t is None:
        print(f"  {name}: None")
        return
    if hasattr(t, 'feats'):
        # SparseTensor
        feats = t.feats
        print(f"  {name} (SparseTensor): feats shape={feats.shape}, "
              f"mean={feats.mean().item():.6f}, std={feats.std().item():.6f}, "
              f"min={feats.min().item():.6f}, max={feats.max().item():.6f}")
    elif torch.is_tensor(t):
        print(f"  {name}: shape={t.shape}, dtype={t.dtype}, "
              f"mean={t.float().mean().item():.6f}, std={t.float().std().item():.6f}, "
              f"min={t.min().item():.6f}, max={t.max().item():.6f}")
    else:
        print(f"  {name}: type={type(t)}")


def compare_tensors(t1, t2, name="tensor", rtol=1e-4, atol=1e-5):
    """对比两个 tensor 是否相近（在 CPU 上对比以避免跨 GPU 问题）"""
    if t1 is None or t2 is None:
        print(f"  {name}: 无法对比 (t1={t1 is not None}, t2={t2 is not None})")
        return False
    
    # 处理 SparseTensor
    if hasattr(t1, 'feats'):
        t1 = t1.feats
    if hasattr(t2, 'feats'):
        t2 = t2.feats
    
    # 移动到 CPU 进行对比
    t1 = t1.detach().cpu()
    t2 = t2.detach().cpu()
    
    if t1.shape != t2.shape:
        print(f"  {name}: 形状不匹配! ref={t1.shape}, ours={t2.shape}")
        return False
    
    diff = (t1.float() - t2.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_close = torch.allclose(t1.float(), t2.float(), rtol=rtol, atol=atol)
    status = "✓ 匹配" if is_close else "✗ 不匹配"
    print(f"  {name}: {status}, max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
    return is_close


def run_reference_pipeline(image: Image.Image, seed: int, model_path: str, gpu_id: int = 0):
    """运行源代码 pipeline，返回中间结果"""
    print("\n" + "="*60)
    print(f"运行 源代码 Pipeline (TrellisImageTo3DPipeline.run) on GPU {gpu_id}")
    print("="*60)
    
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.modules.sparse import SparseTensor
    
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)  # 设置默认 GPU
    pipe = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipe.to(device)
    
    results = {}
    
    # Step 1: Preprocess
    print("\n[Step 1] preprocess_image")
    preprocessed = pipe.preprocess_image(image)
    results['preprocessed'] = preprocessed
    print(f"  preprocessed: size={preprocessed.size}, mode={preprocessed.mode}")
    
    # Step 2: Get cond
    print("\n[Step 2] get_cond")
    cond = pipe.get_cond([preprocessed])
    results['cond'] = cond['cond']
    results['neg_cond'] = cond['neg_cond']
    tensor_stats(cond['cond'], 'cond')
    tensor_stats(cond['neg_cond'], 'neg_cond')
    
    # Step 3: Sample sparse structure
    print("\n[Step 3] sample_sparse_structure")
    torch.manual_seed(seed)
    coords = pipe.sample_sparse_structure(cond, num_samples=1)
    results['coords'] = coords
    tensor_stats(coords, 'coords')
    
    # Step 4: Sample slat (这是关键步骤)
    print("\n[Step 4] sample_slat")
    torch.manual_seed(seed)  # 重置种子以便对比
    slat = pipe.sample_slat(cond, coords)
    results['slat'] = slat
    tensor_stats(slat, 'slat (after normalization)')
    
    # 额外：获取 normalization 参数
    results['slat_normalization'] = pipe.slat_normalization
    print(f"  slat_normalization: mean shape={len(pipe.slat_normalization['mean'])}, "
          f"std shape={len(pipe.slat_normalization['std'])}")
    
    # Step 5: Decode
    print("\n[Step 5] decode_slat")
    outputs = pipe.decode_slat(slat, formats=['mesh'])
    mesh = outputs['mesh'][0]  # 取第一个
    results['mesh'] = mesh
    print(f"  mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
    
    return results, pipe


def run_our_pipeline(image: Image.Image, seed: int, model_path: str, ref_results: dict, gpu_id: int = 1):
    """运行我们的 adapter + rollout，返回中间结果并对比"""
    print("\n" + "="*60)
    print(f"运行 我们的 Pipeline (TrellisRefAdapter + rollout_sparse) on GPU {gpu_id}")
    print("="*60)
    
    from types import SimpleNamespace
    import ml_collections
    
    from edit4shape.generators.trellis.pipeline_adapter import build_pipeline_from_reference
    from edit4shape.systems.trellis import rollout_sparse, TrellisState
    from trellis.modules.sparse import SparseTensor
    
    # 使用指定的 GPU（创建 mock accelerator）
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)  # 设置默认 CUDA 设备
    
    class MockAccelerator:
        def __init__(self, device):
            self._device = device
        @property
        def device(self):
            return self._device
        def is_main_process(self):
            return True
    
    accelerator = MockAccelerator(device)
    device = accelerator.device
    
    # 创建 mock config
    cfg = ml_collections.ConfigDict()
    cfg.seed = seed
    
    # 构建 pipeline adapter
    mock_cfg = SimpleNamespace(
        pretrained=SimpleNamespace(model=model_path),
        verbose=True
    )
    pipeline = build_pipeline_from_reference(mock_cfg, accelerator)
    
    results = {}
    
    # Step 1: Preprocess (使用 adapter 的方法)
    print("\n[Step 1] preprocess_image")
    preprocessed_list = [pipeline.pipe.preprocess_image(image)]
    results['preprocessed'] = preprocessed_list[0]
    print(f"  preprocessed: size={preprocessed_list[0].size}, mode={preprocessed_list[0].mode}")
    
    # Step 2: prepare_image_conditions
    print("\n[Step 2] prepare_image_conditions")
    cond_dict = pipeline.prepare_image_conditions([image])
    results['cond'] = cond_dict['cond']
    results['neg_cond'] = cond_dict['neg_cond']
    tensor_stats(cond_dict['cond'], 'cond')
    tensor_stats(cond_dict['neg_cond'], 'neg_cond')
    
    print("\n  >> 对比 cond:")
    compare_tensors(ref_results['cond'], cond_dict['cond'], 'cond')
    compare_tensors(ref_results['neg_cond'], cond_dict['neg_cond'], 'neg_cond')
    
    # Step 3: dense_sampling
    print("\n[Step 3] dense_sampling")
    torch.manual_seed(seed)
    ss_steps, _, slat_steps, slat_guidance, slat_rescale_t, _ = pipeline.get_sampler_runtime_params()
    print(f"  sampler params: ss_steps={ss_steps}, slat_steps={slat_steps}, "
          f"slat_guidance={slat_guidance}, slat_rescale_t={slat_rescale_t}")
    
    coords = pipeline.dense_sampling(cond_dict, steps=ss_steps)
    results['coords'] = coords
    tensor_stats(coords, 'coords')
    
    print("\n  >> 对比 coords:")
    compare_tensors(ref_results['coords'], coords, 'coords')
    
    # Step 4: rollout_sparse (核心对比)
    print("\n[Step 4] rollout_sparse (对应 sample_slat)")
    
    # 准备 state
    state = TrellisState()
    batch = {
        'pixel_values': [image],
        'image_path': ['test.png'],
        'Conditions': cond_dict,
    }
    state.attach_batch(batch)
    state.coords = coords
    
    # 创建 mock system
    class MockSystem:
        def __init__(self, pipe):
            self.pipeline = pipe
    
    system = MockSystem(pipeline)
    
    # 显式清理 GPU 缓存（dense_sampling 会占用大量中间内存）
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU 缓存已清理，当前显存使用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB")
    
    # 运行 rollout
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    rollout_out = rollout_sparse(state, cfg, system, device, generator=generator, is_training=False)
    slat = rollout_out['latents']
    results['slat'] = slat
    tensor_stats(slat, 'slat (after normalization)')
    
    print("\n  >> 对比 slat:")
    compare_tensors(ref_results['slat'], slat, 'slat')
    
    # 详细对比 slat 的统计信息
    ref_feats = ref_results['slat'].feats
    our_feats = slat.feats
    print(f"\n  详细统计:")
    print(f"    ref slat: mean={ref_feats.mean().item():.6f}, std={ref_feats.std().item():.6f}")
    print(f"    our slat: mean={our_feats.mean().item():.6f}, std={our_feats.std().item():.6f}")
    
    # Step 5: Decode
    print("\n[Step 5] decode")
    outputs = pipeline.decode(slat, formats=['mesh'])
    mesh_list = outputs['mesh']
    # decode 返回的是 list，取第一个
    mesh = mesh_list[0] if isinstance(mesh_list, list) else mesh_list
    results['mesh'] = mesh
    print(f"  mesh: vertices={mesh.vertices.shape}, faces={mesh.faces.shape}")
    
    print("\n  >> 对比 mesh:")
    compare_tensors(ref_results['mesh'].vertices, mesh.vertices, 'vertices')
    compare_tensors(ref_results['mesh'].faces, mesh.faces, 'faces')
    
    return results, pipeline


def save_meshes(ref_results, our_results, output_dir):
    """保存对比的 mesh 文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import trimesh
    
    # 保存参考 mesh
    ref_mesh = ref_results['mesh']
    ref_trimesh = trimesh.Trimesh(
        vertices=ref_mesh.vertices.detach().cpu().numpy(),
        faces=ref_mesh.faces.detach().cpu().numpy(),
        process=False
    )
    ref_path = output_dir / "ref_mesh.obj"
    ref_trimesh.export(str(ref_path))
    print(f"  参考 mesh 保存到: {ref_path}")
    
    # 保存我们的 mesh
    our_mesh = our_results['mesh']
    our_trimesh = trimesh.Trimesh(
        vertices=our_mesh.vertices.detach().cpu().numpy(),
        faces=our_mesh.faces.detach().cpu().numpy(),
        process=False
    )
    our_path = output_dir / "our_mesh.obj"
    our_trimesh.export(str(our_path))
    print(f"  我们的 mesh 保存到: {our_path}")


def save_ref_results(ref_results, output_dir):
    """保存参考结果到文件"""
    import pickle
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 将 tensor 移到 CPU 并保存
    save_data = {}
    for k, v in ref_results.items():
        if hasattr(v, 'feats'):
            # SparseTensor
            save_data[k] = {'type': 'sparse', 'feats': v.feats.detach().cpu(), 'coords': v.coords.detach().cpu()}
        elif torch.is_tensor(v):
            save_data[k] = {'type': 'tensor', 'data': v.detach().cpu()}
        elif hasattr(v, 'vertices'):
            # MeshExtractResult
            save_data[k] = {'type': 'mesh', 'vertices': v.vertices.detach().cpu(), 'faces': v.faces.detach().cpu()}
        else:
            save_data[k] = {'type': 'other', 'data': v}
    
    with open(output_dir / "ref_results.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"  参考结果已保存到 {output_dir / 'ref_results.pkl'}")


def load_ref_results(output_dir):
    """从文件加载参考结果"""
    import pickle
    from trellis.modules.sparse import SparseTensor
    
    output_dir = Path(output_dir)
    with open(output_dir / "ref_results.pkl", "rb") as f:
        save_data = pickle.load(f)
    
    ref_results = {}
    for k, v in save_data.items():
        if v['type'] == 'sparse':
            ref_results[k] = SparseTensor(feats=v['feats'], coords=v['coords'])
        elif v['type'] == 'tensor':
            ref_results[k] = v['data']
        elif v['type'] == 'mesh':
            # 创建一个简单的对象来存储 mesh 数据
            class MeshData:
                pass
            m = MeshData()
            m.vertices = v['vertices']
            m.faces = v['faces']
            ref_results[k] = m
        else:
            ref_results[k] = v['data']
    
    return ref_results


def main():
    args = parse_args()
    
    print("="*60)
    print("Trellis Pipeline 对比调试")
    print("="*60)
    print(f"图片路径: {args.image}")
    print(f"随机种子: {args.seed}")
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行模式: {args.mode}")
    
    # 加载测试图片
    image = Image.open(args.image)
    print(f"图片尺寸: {image.size}, 模式: {image.mode}")
    
    output_dir = Path(args.output_dir)
    
    if args.mode in ["ref", "both"]:
        # 运行参考 pipeline
        ref_results, ref_pipe = run_reference_pipeline(image, args.seed, args.model_path, gpu_id=args.ref_gpu)
        save_ref_results(ref_results, output_dir)
        
        # 释放内存
        del ref_pipe
        del ref_results
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("\n已释放参考 pipeline 内存")
        
        if args.mode == "ref":
            print("\n仅运行参考代码完成。请使用 --mode ours 运行对比。")
            return
    
    if args.mode in ["ours", "both"]:
        # 加载参考结果
        ref_results = load_ref_results(output_dir)
        print(f"\n已加载参考结果")
        
        # 运行我们的 pipeline
        our_results, our_pipe = run_our_pipeline(image, args.seed, args.model_path, ref_results, gpu_id=args.our_gpu)
        
        # 保存 mesh 文件
        print("\n" + "="*60)
        print("保存对比结果")
        print("="*60)
        save_meshes(ref_results, our_results, args.output_dir)
        
        print("\n" + "="*60)
        print("对比完成!")
        print("="*60)


if __name__ == "__main__":
    main()


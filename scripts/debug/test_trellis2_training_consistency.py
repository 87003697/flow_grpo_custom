"""
Trellis2 训练采样一致性测试 (串行可靠版)
=========================================

验证方式：在同一个 GPU 上串行运行两条路径，每条路径运行前重置随机种子，
确保两条路径使用完全相同的随机数序列。

对比两条路径：
- 路径 A：调用官方基类方法 (sample_sparse_structure, sample_shape_slat_cascade, sample_tex_slat)
- 路径 B：调用训练脚本使用的封装方法 (stage_1, stage_2_shape_cascade, stage_2_tex)
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import os
import traceback
import pickle
import gc

# --- 环境配置 ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
ref_root = ROOT / "_reference_codes" / "TRELLIS.2"
sys.path.append(str(ref_root))
sys.path.append(str(ref_root / "o-voxel"))

# 设置环境变量
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_all_seeds(seed: int, device: torch.device):
    """设置所有随机种子，确保完全可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sparse_to_dict(sp):
    """将 SparseTensor 转换为可序列化的字典 (CPU tensors)"""
    return {
        'feats': sp.feats.detach().cpu(),
        'coords': sp.coords.detach().cpu(),
    }


def get_pipeline(device):
    """加载 Pipeline 并移动到指定设备"""
    from flow_grpo.diffusers_patch.trellis2_pipeline_with_logprob import Trellis2PipelineWithLogProb
    
    dino_local = ref_root / "pretrained_weights" / "dinov3-vitl16-pretrain-lvd1689m" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    pipeline_path = "microsoft/TRELLIS.2-4B"
    
    print(f"[{device}] Loading pipeline...")
    pipeline = Trellis2PipelineWithLogProb.from_pretrained(
        pipeline_path,
        dino_local_path=str(dino_local) if dino_local.exists() else None
    )
    
    # 禁用 low_vram 模式，确保所有模型在同一设备上
    pipeline.low_vram = False
    pipeline.to(device)
        
    return pipeline


def run_path_A(pipeline, image_proc, cond_512_dict, cond_1024_dict, device, seed):
    """
    路径 A：调用官方基类方法
    """
    print(f"\n{'='*50}")
    print("Running Path A: Official base class methods")
    print('='*50)
    
    # 重置随机种子
    set_all_seeds(seed, device)
    
    captures = {}
    captures['cond_512'] = cond_512_dict['cond'].detach().cpu()
    captures['cond_1024'] = cond_1024_dict['cond'].detach().cpu()
    
    # Stage 1: sample_sparse_structure (官方方法)
    print("Stage 1: sample_sparse_structure...")
    coords = pipeline.sample_sparse_structure(
        cond_512_dict,
        resolution=32,
        num_samples=1
    )
    captures['stage1_coords'] = coords.detach().cpu()
    print(f"  -> coords shape: {coords.shape}")
    
    # Stage 2 Shape (Cascade): sample_shape_slat_cascade (官方方法)
    print("Stage 2 Shape: sample_shape_slat_cascade...")
    shape_slat_hr, actual_res = pipeline.sample_shape_slat_cascade(
        lr_cond=cond_512_dict,
        cond=cond_1024_dict,
        flow_model_lr=pipeline.models['shape_slat_flow_model_512'],
        flow_model=pipeline.models['shape_slat_flow_model_1024'],
        lr_resolution=512,
        resolution=1024,
        coords=coords
    )
    captures['shape_slat_1024'] = sparse_to_dict(shape_slat_hr)
    captures['resolution'] = actual_res
    print(f"  -> shape_slat coords: {shape_slat_hr.coords.shape}, resolution: {actual_res}")
    
    # Stage 2 Tex: sample_tex_slat (官方方法)
    print("Stage 2 Tex: sample_tex_slat...")
    tex_slat = pipeline.sample_tex_slat(
        cond=cond_1024_dict,
        flow_model=pipeline.models['tex_slat_flow_model_1024'],
        shape_slat=shape_slat_hr
    )
    captures['tex_slat'] = sparse_to_dict(tex_slat)
    print(f"  -> tex_slat coords: {tex_slat.coords.shape}")
    
    # Decode Mesh: decode_latent (官方方法)
    print("Decoding mesh...")
    meshes = pipeline.decode_latent(shape_slat_hr, tex_slat, actual_res)
    mesh = meshes[0]
    captures['mesh_vertices'] = mesh.vertices.clone().detach().cpu().float()
    captures['mesh_faces'] = mesh.faces.clone().detach().cpu().long()
    print(f"  -> mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")
    
    print("Path A completed.\n")
    return captures


def run_path_B(pipeline, image_proc, cond_512_dict, cond_1024_dict, device, seed):
    """
    路径 B：调用训练脚本封装方法
    """
    print(f"\n{'='*50}")
    print("Running Path B: Training script wrapper methods")
    print('='*50)
    
    # 重置随机种子 (与路径 A 相同)
    set_all_seeds(seed, device)
    
    captures = {}
    captures['cond_512'] = cond_512_dict['cond'].detach().cpu()
    captures['cond_1024'] = cond_1024_dict['cond'].detach().cpu()
    
    # Stage 1: stage_1 (封装方法)
    print("Stage 1: stage_1...")
    coords_B, _ = pipeline.stage_1(
        cond=cond_512_dict,
        ss_resolution=32,
        num_samples=1
    )
    captures['stage1_coords'] = coords_B.detach().cpu()
    print(f"  -> coords shape: {coords_B.shape}")
    
    # Stage 2 Shape (Cascade): stage_2_shape_cascade (封装方法)
    print("Stage 2 Shape: stage_2_shape_cascade...")
    shape_slat_B, shape_slat_512_B, res_B = pipeline.stage_2_shape_cascade(
        lr_cond=cond_512_dict,
        hr_cond=cond_1024_dict,
        coords=coords_B,
        lr_resolution=512,
        hr_resolution=1024
    )
    captures['shape_slat_1024'] = sparse_to_dict(shape_slat_B)
    captures['shape_slat_512'] = sparse_to_dict(shape_slat_512_B)
    captures['resolution'] = res_B
    print(f"  -> shape_slat coords: {shape_slat_B.coords.shape}, resolution: {res_B}")
    
    # Stage 2 Tex: stage_2_tex (封装方法)
    print("Stage 2 Tex: stage_2_tex...")
    tex_slat_B, _ = pipeline.stage_2_tex(
        cond=cond_1024_dict,
        shape_slat=shape_slat_B
    )
    captures['tex_slat'] = sparse_to_dict(tex_slat_B)
    print(f"  -> tex_slat coords: {tex_slat_B.coords.shape}")
    
    # Decode Mesh: export_mesh (封装方法)
    print("Decoding mesh...")
    mesh_B = pipeline.export_mesh(
        shape_slat=shape_slat_B,
        tex_slat=tex_slat_B,
        resolution=res_B
    )
    captures['mesh_vertices'] = mesh_B.vertices.clone().detach().cpu().float()
    captures['mesh_faces'] = mesh_B.faces.clone().detach().cpu().long()
    print(f"  -> mesh: {mesh_B.vertices.shape[0]} verts, {mesh_B.faces.shape[0]} faces")
    
    print("Path B completed.\n")
    return captures


def assert_close(name, a, b, atol=1e-5):
    """断言两个值是否接近"""
    if isinstance(a, dict) and 'feats' in a:
        # SparseTensor 比较
        if not torch.equal(a['coords'], b['coords']):
            print(f"❌ {name} Coords Mismatch!")
            print(f"   A coords shape: {a['coords'].shape}, B coords shape: {b['coords'].shape}")
            # 显示前几个坐标的差异
            min_len = min(len(a['coords']), len(b['coords']))
            if min_len > 0:
                diff_count = (a['coords'][:min_len] != b['coords'][:min_len]).any(dim=1).sum().item()
                print(f"   First {min_len} coords: {diff_count} differences")
            return False
        return assert_close(f"{name} Feats", a['feats'], b['feats'], atol)
    elif isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            print(f"❌ {name} Shape Mismatch! {a.shape} != {b.shape}")
            return False
        if not torch.allclose(a.float(), b.float(), atol=atol):
            diff = (a.float() - b.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"❌ {name} Value Mismatch! Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
            return False
    else:
        if a != b:
            print(f"❌ {name} Mismatch! {a} != {b}")
            return False
            
    print(f"✅ {name} Match")
    return True


def main():
    device = torch.device('cuda:0')
    seed = 42
    
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载 Pipeline (只加载一次，两条路径共用)
    pipeline = get_pipeline(device)
    
    # 加载图片
    image_path = ref_root / "assets" / "example_image" / "T.png"
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    print(f"Image: {image_path}")
    
    image = Image.open(image_path)
    
    # 预处理图像 (只做一次)
    print("Preprocessing image...")
    image_proc = pipeline.preprocess_image(image)
    
    # 准备条件 (只做一次，两条路径共用相同的条件)
    # 这一步在种子重置之前完成，不影响后续采样的随机性
    print("Preparing conditions...")
    cond_512_dict = pipeline.get_cond([image_proc], 512)
    cond_1024_dict = pipeline.get_cond([image_proc], 1024)
    
    # =========================================================================
    # 运行路径 A (官方方法)
    # =========================================================================
    res_A = run_path_A(pipeline, image_proc, cond_512_dict, cond_1024_dict, device, seed)
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # =========================================================================
    # 运行路径 B (封装方法)
    # =========================================================================
    res_B = run_path_B(pipeline, image_proc, cond_512_dict, cond_1024_dict, device, seed)
    
    # =========================================================================
    # 对比结果
    # =========================================================================
    print("\n" + "="*60)
    print("CONSISTENCY CHECK REPORT (Serial Mode - Same GPU, Same Seed)")
    print("="*60)
    print("Path A: Official base class methods")
    print("Path B: Training script wrapper methods")
    print("="*60)
    
    all_passed = True
    
    # 1. Conditioning
    print("\n--- 1. Image Conditioning ---")
    all_passed &= assert_close("Cond 512", res_A['cond_512'], res_B['cond_512'])
    all_passed &= assert_close("Cond 1024", res_A['cond_1024'], res_B['cond_1024'])
    
    # 2. Stage 1 Coords
    print("\n--- 2. Stage 1 (Sparse Structure) ---")
    coords_match = assert_close("Stage 1 Coords", res_A['stage1_coords'], res_B['stage1_coords'])
    all_passed &= coords_match
    
    # 3. Stage 2 Shape
    print("\n--- 3. Stage 2 Shape (1024 HR) ---")
    if coords_match:
        all_passed &= assert_close("Shape Latent 1024", res_A['shape_slat_1024'], res_B['shape_slat_1024'])
    else:
        print("⏭️  Skipped (Stage 1 coords mismatch)")
    all_passed &= assert_close("Resolution", res_A['resolution'], res_B['resolution'])
    
    # 4. Stage 2 Tex
    print("\n--- 4. Stage 2 Texture ---")
    if coords_match:
        all_passed &= assert_close("Texture Latent", res_A['tex_slat'], res_B['tex_slat'])
    else:
        print("⏭️  Skipped (Stage 1 coords mismatch)")
    
    # 5. Mesh
    print("\n--- 5. Final Mesh ---")
    print(f"Mesh A: {res_A['mesh_vertices'].shape[0]} verts, {res_A['mesh_faces'].shape[0]} faces")
    print(f"Mesh B: {res_B['mesh_vertices'].shape[0]} verts, {res_B['mesh_faces'].shape[0]} faces")
    if coords_match:
        all_passed &= assert_close("Mesh Vertices", res_A['mesh_vertices'], res_B['mesh_vertices'])
        all_passed &= assert_close("Mesh Faces", res_A['mesh_faces'], res_B['mesh_faces'])
    else:
        print("⏭️  Skipped (Stage 1 coords mismatch)")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("Training wrapper methods are 100% consistent with official methods.")
    else:
        print("⚠️  SOME CHECKS FAILED!")
        print("Please investigate the differences between wrapper and official methods.")
    print("="*60)


if __name__ == "__main__":
    main()

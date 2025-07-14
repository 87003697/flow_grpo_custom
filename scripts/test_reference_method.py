#!/usr/bin/env python3
"""
使用参考代码的标准方法测试
对比自定义SDE实现和标准scheduler.step方法的差异
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from generators.hunyuan3d.hy3dshape.pipelines import retrieve_timesteps


def test_reference_method():
    """使用参考代码的标准方法测试"""
    print("🧪 使用参考代码的标准方法测试")
    print("=" * 60)
    
    # 1. 初始化Pipeline
    print("\n🎯 步骤 1: 初始化Pipeline")
    try:
        wrapper_pipeline = Hunyuan3DPipeline()
        pipeline = wrapper_pipeline.core_pipeline
        print("  ✅ Pipeline 初始化成功")
    except Exception as e:
        print(f"  ❌ Pipeline 初始化失败: {e}")
        return False
    
    # 2. 加载测试图像
    print("\n🎯 步骤 2: 加载测试图像")
    test_image_path = "dataset/eval3d/images/walking_siamese_cat.png"
    
    if not os.path.exists(test_image_path):
        print(f"  ❌ 测试图像不存在: {test_image_path}")
        return False
    
    image = Image.open(test_image_path).convert("RGBA")
    print(f"  ✅ 成功加载图像: {test_image_path}")
    
    # 3. 使用参考代码的标准方法
    print("\n🎯 步骤 3: 使用参考代码的标准方法")
    
    # 参考代码的参数
    num_inference_steps = 50
    guidance_scale = 5.0
    octree_resolution = 384
    mc_level = 0.0
    num_chunks = 8000
    
    device = pipeline.device
    dtype = pipeline.dtype
    
    # 🔧 关键：使用参考代码的条件处理方式
    do_classifier_free_guidance = guidance_scale >= 0 and not (
        hasattr(pipeline.model, 'guidance_embed') and
        pipeline.model.guidance_embed is True
    )
    
    cond_inputs = pipeline.prepare_image(image)
    image_tensor = cond_inputs.pop('image')
    cond = pipeline.encode_cond(
        image=image_tensor,
        additional_cond_inputs=cond_inputs,
        do_classifier_free_guidance=do_classifier_free_guidance,
        dual_guidance=False,
    )
    
    batch_size = image_tensor.shape[0]
    
    # 🔧 关键：使用参考代码的timesteps处理
    sigmas = np.linspace(0, 1, num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
    )
    
    latents = pipeline.prepare_latents(batch_size, dtype, device, torch.Generator().manual_seed(42))
    
    # 🔧 关键：使用参考代码的guidance处理
    guidance = None
    if hasattr(pipeline.model, 'guidance_embed') and pipeline.model.guidance_embed is True:
        guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
    
    print(f"  📊 初始latents范围: [{latents.min():.6f}, {latents.max():.6f}]")
    
    # 4. 使用标准的scheduler.step方法进行扩散采样
    print("\n🎯 步骤 4: 使用标准的scheduler.step方法")
    
    start_time = time.time()
    
    for i, t in enumerate(tqdm(timesteps, desc="标准方法扩散采样")):
        # 🔧 关键：完全按照参考代码的方式
        if do_classifier_free_guidance:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents
        
        # 🔧 关键：按照参考代码的timestep处理
        timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
        timestep = timestep / pipeline.scheduler.config.num_train_timesteps
        
        # 模型预测
        noise_pred = pipeline.model(latent_model_input, timestep, cond, guidance=guidance)
        
        # 🔧 关键：CFG处理
        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 🔧 关键：使用标准的scheduler.step，不是自定义SDE
        outputs = pipeline.scheduler.step(noise_pred, t, latents)
        latents = outputs.prev_sample
        
        # 每10步打印一次状态
        if (i + 1) % 10 == 0:
            print(f"    步骤 {i+1:2d}: latents范围 [{latents.min():.6f}, {latents.max():.6f}]")
    
    end_time = time.time()
    print(f"  ✅ 扩散采样完成，耗时: {end_time - start_time:.2f}秒")
    print(f"  📊 最终latents范围: [{latents.min():.6f}, {latents.max():.6f}]")
    
    # 5. 使用标准的VAE解码和mesh生成
    print("\n🎯 步骤 5: 使用标准的VAE解码和mesh生成")
    
    # 🔧 关键：完全按照参考代码的_export方法
    latents = 1. / pipeline.vae.scale_factor * latents
    latents = pipeline.vae(latents)
    
    print(f"  📊 VAE解码后latents范围: [{latents.min():.6f}, {latents.max():.6f}]")
    
    # 🔧 关键：捕获grid_logits
    grid_logits = pipeline.vae.decoder(latents)
    print(f"  📊 Grid Logits范围: [{grid_logits.min():.6f}, {grid_logits.max():.6f}]")
    
    # 检查是否解决了全负值问题
    has_positive = (grid_logits > 0).any().item()
    has_negative = (grid_logits < 0).any().item()
    
    print(f"  🔍 Grid Logits分析:")
    print(f"    包含正值: {has_positive}")
    print(f"    包含负值: {has_negative}")
    print(f"    数值健康: {not torch.isnan(grid_logits).any().item() and not torch.isinf(grid_logits).any().item()}")
    
    if has_positive and has_negative:
        print(f"  ✅ Grid Logits范围正常！包含正负值")
        success = True
    else:
        print(f"  ❌ Grid Logits范围异常！仍然是全正或全负")
        success = False
    
    # 6. 尝试生成mesh
    print("\n🎯 步骤 6: 尝试生成mesh")
    
    try:
        mesh_output = pipeline.vae.latents2mesh(
            latents,
            bounds=1.01,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=None,
            enable_pbar=True,
        )
        
        # 转换为trimesh
        from generators.hunyuan3d.hy3dshape.pipelines import export_to_trimesh
        meshes = export_to_trimesh(mesh_output)
        mesh = meshes[0] if isinstance(meshes, list) else meshes
        
        # 保存mesh
        output_path = "output_reference_method.obj"
        mesh.export(output_path)
        
        file_size = os.path.getsize(output_path)
        print(f"  ✅ Mesh生成成功！")
        print(f"    文件: {output_path} ({file_size / (1024*1024):.2f} MB)")
        print(f"    顶点数: {len(mesh.vertices)}")
        print(f"    面数: {len(mesh.faces)}")
        
        # 7. 渲染mesh
        print("\n🎯 步骤 7: 渲染mesh")
        try:
            from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import simple_render_mesh
            render_output = "output_reference_method_render.png"
            simple_render_mesh(output_path, render_output)
            
            if os.path.exists(render_output):
                render_size = os.path.getsize(render_output)
                print(f"  ✅ 渲染成功: {render_output} ({render_size / 1024:.1f} KB)")
            else:
                print(f"  ❌ 渲染失败")
                
        except Exception as e:
            print(f"  ⚠️ 渲染失败: {e}")
        
    except Exception as e:
        print(f"  ❌ Mesh生成失败: {e}")
        success = False
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 测试成功！标准方法解决了grid_logits问题")
        print("💡 结论: 问题出在自定义SDE实现上，标准scheduler.step方法正常工作")
        print("🔧 建议: 修复或替换自定义SDE实现")
    else:
        print("❌ 测试失败！问题可能不在SDE实现上")
        print("🔍 需要进一步调查其他可能的原因")
    
    return success


def main():
    """主函数"""
    print("🧪 测试参考代码的标准方法")
    print("=" * 80)
    
    result = test_reference_method()
    
    if result:
        print("\n🎉 测试完成！发现问题根源")
        return 0
    else:
        print("\n❌ 测试失败！需要进一步调查")
        return 1


if __name__ == "__main__":
    exit(main()) 
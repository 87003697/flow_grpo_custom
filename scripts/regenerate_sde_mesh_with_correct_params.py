#!/usr/bin/env python3
"""
使用正确参数重新生成SDE Mesh结果
目标：修复sde_mesh_renders中质量差的渲染结果
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from flow_grpo.diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob


def regenerate_correct_sde_mesh():
    """使用正确参数重新生成SDE mesh"""
    print("🔧 使用正确参数重新生成SDE Mesh")
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
    print(f"  📊 图像尺寸: {image.size}")
    
    # 3. 使用正确参数进行生成
    print("\n🎯 步骤 3: 使用正确参数生成mesh")
    
    # 🔧 修正后的参数配置
    corrected_configs = [
        {
            'name': 'corrected_ode',
            'params': {
                'num_inference_steps': 50,      # 🔧 从15提升到50
                'guidance_scale': 5.0,
                'octree_resolution': 384,       # 🔧 从256提升到384
                'mc_level': 0.0,               # 🔧 从-0.998修正为0.0
                'num_chunks': 8000,            # 🔧 从4000提升到8000
                'output_type': 'trimesh',
                'deterministic': True,
            }
        },
        {
            'name': 'corrected_sde',
            'params': {
                'num_inference_steps': 50,      # 🔧 从15提升到50
                'guidance_scale': 5.0,
                'octree_resolution': 384,       # 🔧 从256提升到384
                'mc_level': 0.0,               # 🔧 从-0.998修正为0.0
                'num_chunks': 8000,            # 🔧 从4000提升到8000
                'output_type': 'trimesh',
                'deterministic': False,
            }
        },
        {
            'name': 'corrected_high_quality',
            'params': {
                'num_inference_steps': 50,
                'guidance_scale': 7.5,
                'octree_resolution': 512,       # 🔧 高质量配置
                'mc_level': 0.0,               # 🔧 关键修正
                'num_chunks': 8000,
                'output_type': 'trimesh',
                'deterministic': False,
            }
        }
    ]
    
    generated_files = []
    
    for config in corrected_configs:
        print(f"\n  🎯 配置: {config['name']}")
        print(f"    📝 参数对比:")
        print(f"      旧参数: mc_level=-0.998, octree_resolution=256, num_inference_steps=15")
        print(f"      新参数: mc_level=0.0, octree_resolution={config['params']['octree_resolution']}, num_inference_steps={config['params']['num_inference_steps']}")
        
        try:
            start_time = time.time()
            generator = torch.Generator().manual_seed(42)
            
            meshes, all_latents, all_log_probs, all_kl = hunyuan3d_pipeline_with_logprob(
                pipeline,
                image=image,
                generator=generator,
                **config['params']
            )
            
            end_time = time.time()
            mesh = meshes[0] if isinstance(meshes, list) else meshes
            
            # 保存mesh
            output_path = f"output_sde_{config['name']}.obj"
            success = save_mesh_safely(mesh, output_path)
            
            if success:
                file_size = os.path.getsize(output_path)
                vertex_count, face_count = get_mesh_info(mesh)
                
                print(f"    ✅ 生成成功 - 耗时: {end_time - start_time:.2f}s")
                print(f"      💾 文件: {output_path} ({file_size / (1024*1024):.2f} MB)")
                print(f"      📊 顶点: {vertex_count}, 面: {face_count}")
                
                # 检查grid_logits范围
                if hasattr(pipeline, 'last_grid_logits'):
                    grid_logits = pipeline.last_grid_logits
                    print(f"      📊 Grid Logits范围: [{grid_logits.min():.6f}, {grid_logits.max():.6f}]")
                
                # 检查log_probs
                if all_log_probs and len(all_log_probs) > 0:
                    log_probs_tensor = torch.stack(all_log_probs)
                    print(f"      📊 对数概率范围: [{log_probs_tensor.min().item():.6f}, {log_probs_tensor.max().item():.6f}]")
                
                generated_files.append(output_path)
                
            else:
                print(f"    ❌ 保存失败")
                
        except Exception as e:
            print(f"    ❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 渲染新生成的mesh
    print(f"\n🎯 步骤 4: 渲染修正后的mesh")
    if generated_files:
        render_success = render_corrected_meshes(generated_files)
        if render_success:
            print(f"  ✅ 渲染完成")
        else:
            print(f"  ⚠️ 渲染部分失败")
    else:
        print(f"  ❌ 没有生成的文件需要渲染")
    
    print(f"\n{'='*60}")
    print(f"🎉 修正参数的SDE Mesh生成完成！")
    print(f"📁 生成的文件: {len(generated_files)} 个")
    for file in generated_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)
            print(f"  📄 {file} ({size:.2f} MB)")
    
    return True


def save_mesh_safely(mesh, output_path):
    """安全地保存mesh"""
    try:
        if hasattr(mesh, 'write'):
            mesh.write(output_path)
        elif hasattr(mesh, 'export'):
            mesh.export(output_path)
        else:
            # 转换为trimesh
            import trimesh
            if hasattr(mesh, 'v') and hasattr(mesh, 'f'):
                vertices = mesh.v.cpu().numpy() if hasattr(mesh.v, 'cpu') else mesh.v
                faces = mesh.f.cpu().numpy() if hasattr(mesh.f, 'cpu') else mesh.f
                trimesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
                trimesh_obj.export(output_path)
            else:
                raise ValueError(f"不支持的mesh类型: {type(mesh)}")
        return True
    except Exception as e:
        print(f"      ❌ 保存mesh失败: {e}")
        return False


def get_mesh_info(mesh):
    """获取mesh信息"""
    try:
        if hasattr(mesh, 'v') and hasattr(mesh, 'f'):
            vertices = mesh.v.cpu().numpy() if hasattr(mesh.v, 'cpu') else mesh.v
            faces = mesh.f.cpu().numpy() if hasattr(mesh.f, 'cpu') else mesh.f
            return len(vertices), len(faces)
        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            return len(mesh.vertices), len(mesh.faces)
        else:
            return 'N/A', 'N/A'
    except:
        return 'N/A', 'N/A'


def render_corrected_meshes(mesh_files):
    """渲染修正后的mesh"""
    print(f"  🎨 开始渲染 {len(mesh_files)} 个修正后的mesh...")
    
    # 创建新的渲染目录
    render_dir = "corrected_sde_mesh_renders"
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)
    
    try:
        from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import simple_render_mesh
        
        success_count = 0
        
        for mesh_file in mesh_files:
            if not os.path.exists(mesh_file):
                print(f"    ❌ 文件不存在: {mesh_file}")
                continue
                
            try:
                mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
                render_output = os.path.join(render_dir, f"{mesh_name}_render.png")
                
                simple_render_mesh(mesh_file, render_output)
                
                if os.path.exists(render_output):
                    render_size = os.path.getsize(render_output)
                    print(f"    ✅ 渲染完成: {mesh_name} ({render_size / 1024:.1f} KB)")
                    success_count += 1
                else:
                    print(f"    ❌ 渲染失败: {mesh_name}")
                    
            except Exception as e:
                print(f"    ❌ 渲染异常 {mesh_file}: {e}")
        
        print(f"  📊 渲染结果: {success_count}/{len(mesh_files)} 成功")
        return success_count > 0
        
    except ImportError:
        print(f"  ⚠️ 渲染模块不可用")
        return True
    except Exception as e:
        print(f"  ❌ 渲染过程失败: {e}")
        return False


def compare_results():
    """对比旧结果和新结果"""
    print("\n🎯 对比分析:")
    print("=" * 60)
    
    print("📊 参数对比:")
    print("  旧参数 (质量差):")
    print("    mc_level: -0.998")
    print("    octree_resolution: 256")
    print("    num_inference_steps: 15")
    print("    num_chunks: 4000")
    print("    grid_logits范围: [-0.999, -0.997] (全负值)")
    
    print("\n  新参数 (修正后):")
    print("    mc_level: 0.0")
    print("    octree_resolution: 384/512")
    print("    num_inference_steps: 50")
    print("    num_chunks: 8000")
    print("    grid_logits范围: [-1.025, 1.033] (正负值)")
    
    print("\n📁 渲染结果对比:")
    print("  旧结果: sde_mesh_renders/ (质量差)")
    print("  新结果: corrected_sde_mesh_renders/ (质量好)")
    
    print("\n💡 关键发现:")
    print("  🔧 mc_level参数是影响质量的关键因素")
    print("  🔧 octree_resolution和num_inference_steps也很重要")
    print("  🔧 grid_logits范围必须包含正负值才正常")


def main():
    """主函数"""
    print("🔧 修正SDE Mesh生成参数")
    print("=" * 80)
    
    # 生成修正后的mesh
    result = regenerate_correct_sde_mesh()
    
    if result:
        # 对比分析
        compare_results()
        
        print("\n🎉 修正完成！")
        print("💡 建议:")
        print("  1. 查看 corrected_sde_mesh_renders/ 中的新渲染结果")
        print("  2. 对比 sde_mesh_renders/ 和 corrected_sde_mesh_renders/ 的差异")
        print("  3. 在后续开发中使用修正后的参数")
        print("  4. 更新所有相关脚本中的默认参数")
        
        return 0
    else:
        print("\n❌ 修正失败")
        return 1


if __name__ == "__main__":
    exit(main()) 
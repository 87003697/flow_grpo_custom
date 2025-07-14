#!/usr/bin/env python3
"""
测试修改后的pipeline，使用标准scheduler.step方法
"""

import os
import sys
import torch
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.hunyuan3d.pipeline import Hunyuan3DPipeline
from flow_grpo.diffusers_patch.hunyuan3d_pipeline_with_logprob import hunyuan3d_pipeline_with_logprob


def test_standard_scheduler():
    """测试标准scheduler方法"""
    print("🧪 测试标准scheduler方法")
    print("=" * 50)
    
    # 1. 初始化Pipeline
    print("\n1. 初始化Pipeline...")
    wrapper_pipeline = Hunyuan3DPipeline()
    pipeline = wrapper_pipeline.core_pipeline
    print("✅ Pipeline初始化成功")
    
    # 2. 加载图像
    print("\n2. 加载图像...")
    image_path = "dataset/eval3d/images/walking_siamese_cat.png"
    image = Image.open(image_path).convert("RGBA")
    print(f"✅ 图像加载成功: {image.size}")
    
    # 3. 使用标准scheduler方法生成
    print("\n3. 使用标准scheduler方法生成...")
    
    try:
        meshes, all_latents, all_log_probs, all_kl = hunyuan3d_pipeline_with_logprob(
            pipeline,
            image=image,
            num_inference_steps=20,  # 减少步数以节省时间
            guidance_scale=5.0,
            generator=torch.Generator().manual_seed(42),
            output_type='trimesh',
            octree_resolution=256,  # 减少分辨率以节省内存
            mc_level=0.0,
            num_chunks=4000,  # 减少chunks以节省内存
            deterministic=True,
            use_standard_scheduler=True,  # 🔧 关键：使用标准方法
        )
        
        print("✅ 生成成功！")
        print(f"📊 结果统计:")
        print(f"  all_latents: {len(all_latents)} 个")
        print(f"  all_log_probs: {len(all_log_probs)} 个")
        print(f"  all_kl: {len(all_kl)} 个")
        
        # 4. 保存结果
        mesh = meshes[0] if isinstance(meshes, list) else meshes
        output_path = "test_standard_scheduler.obj"
        
        if hasattr(mesh, 'export'):
            mesh.export(output_path)
        elif hasattr(mesh, 'write'):
            mesh.write(output_path)
        else:
            print("⚠️ 无法保存mesh，格式不支持")
            return False
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Mesh保存成功: {output_path} ({file_size / (1024*1024):.2f} MB)")
            
            # 5. 尝试渲染
            print("\n5. 尝试渲染...")
            try:
                from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import simple_render_mesh
                render_path = "test_standard_scheduler_render.png"
                simple_render_mesh(output_path, render_path)
                
                if os.path.exists(render_path):
                    render_size = os.path.getsize(render_path)
                    print(f"✅ 渲染成功: {render_path} ({render_size / 1024:.1f} KB)")
                    return True
                else:
                    print("⚠️ 渲染失败，但mesh生成成功")
                    return True
                    
            except Exception as e:
                print(f"⚠️ 渲染失败: {e}")
                return True  # mesh生成成功就算成功
        else:
            print("❌ Mesh保存失败")
            return False
            
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🔧 测试标准scheduler方法")
    print("=" * 60)
    
    success = test_standard_scheduler()
    
    if success:
        print("\n🎉 测试成功！")
        print("✅ 标准scheduler方法工作正常")
        print("🔧 建议：在GRPO训练中使用use_standard_scheduler=True")
    else:
        print("\n❌ 测试失败！")
        print("🔍 需要进一步调试")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 
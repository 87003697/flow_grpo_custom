#!/usr/bin/env python3
"""
TRELLIS官方模式使用示例 + Mesh多视角可视化
展示如何正确使用下载的TRELLIS-image-large模型并生成mesh的多视角渲染
"""

import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("🎯 TRELLIS官方模式使用示例 + Mesh可视化")
    print("=" * 60)
    
    # 1. 初始化pipeline
    print("🔄 正在初始化TRELLIS Pipeline...")
    from generators.trellis.pipeline import TrellisStage2Pipeline
    
    pipeline = TrellisStage2Pipeline()
    print(f"📍 初始设备: {pipeline.device}")
    
    # 2. 转换到GPU (遵循官方模式)
    if torch.cuda.is_available():
        print("🚀 将模型移动到GPU...")
        pipeline.cuda()
        print(f"✅ 设备转换完成: {pipeline.device}")
    
    # 3. 准备测试图像
    print("\n🖼️ 准备测试图像...")
    # 使用指定的真实图片
    image_path = "dataset/eval3d/images/feeding_squirrel.png"
    test_image = Image.open(image_path)
    print(f"图片路径: {image_path}")
    print(f"原始图像尺寸: {test_image.size}")
    print(f"图像模式: {test_image.mode}")
    
    # 4. 使用training模式 (遵循官方模式的上下文管理器)
    print("\n🎯 进入训练模式...")
    with pipeline.train_mode():
        print("✅ 已进入训练模式，Stage 2(SLatFlowModel)可训练")
        
        # 5. 图像预处理和条件编码
        print("\n📸 图像预处理和条件编码...")
        # 预处理图像
        from generators.trellis.utils import trellis_preprocess_image
        preprocessed_image = trellis_preprocess_image(test_image)
        
        # 编码图像条件
        image_conds = pipeline.prepare_image_conditions([preprocessed_image])
        print(f"条件编码成功: {image_conds['cond'].shape}")
        
        # 6. Stage 1推理 (固定，用于获取稀疏结构)
        print("\n🏗️ Stage 1推理 (稀疏结构生成)...")
        coords = pipeline.forward_stage1(image_conds)
        print(f"稀疏坐标: {coords.shape}, 点数: {coords.shape[0]}")
        
        # 7. Stage 2推理 (SLAT生成，这部分将来会训练)
        print("\n🧩 Stage 2推理 (SLAT生成)...")
        slat_output, _, _ = pipeline.forward_stage2_with_logprob(coords, image_conds)
        print(f"SLAT坐标: {slat_output.coords.shape}")
        print(f"SLAT特征: {slat_output.feats.shape}")
    
    # 8. 切换到评估模式进行mesh解码
    print("\n🔍 切换到评估模式...")
    with pipeline.eval_mode():
        print("✅ 已进入评估模式")
        
        # 9. 解码为mesh
        print("\n🎲 解码为mesh...")
        meshes = pipeline.decode_slat_to_mesh(slat_output)
        
        if meshes and len(meshes) > 0:
            mesh = meshes[0]
            print(f"生成mesh: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面")
            
            # 10. Mesh多视角可视化
            print("\n🎨 开始Mesh多视角可视化...")
            
            # 导入扩展的渲染器
            from generators.hunyuan3d.hy3dshape.utils.visualizers.renderer import render_mesh_multiple_views
            
            # 创建输出目录
            output_dir = Path("outputs/trellis_renders/feeding_squirrel")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # 渲染多个预设视角
                presets = ["turntable", "around", "corners"]
                
                for preset in presets:
                    print(f"\n📷 渲染 {preset} 视角...")
                    
                    # 使用trimesh对象直接渲染
                    save_path = output_dir / f"trellis_mesh_{preset}.png"
                    
                    result_path = render_mesh_multiple_views(
                        mesh_trimesh=mesh,
                        save_path=str(save_path),
                        preset=preset,
                        device=pipeline.device.type
                    )
                    
                    print(f"✅ {preset} 视角渲染完成: {result_path}")
                
                # 保存原始mesh文件
                try:
                    # 将MeshExtractResult转换为trimesh
                    from generators.trellis.utils import convert_trellis_to_trimesh
                    trimesh_objects = convert_trellis_to_trimesh([mesh])
                    
                    if trimesh_objects and len(trimesh_objects) > 0:
                        mesh_path = output_dir / "trellis_generated_mesh.obj"
                        trimesh_objects[0].export(str(mesh_path))
                        print(f"\n💾 原始mesh已保存: {mesh_path}")
                    else:
                        print(f"\n⚠️ mesh转换失败，无法保存文件")
                except Exception as e:
                    print(f"\n⚠️ mesh保存失败: {e}")
                    print(f"   mesh类型: {type(mesh)}")
                    if hasattr(mesh, 'vertices'):
                        print(f"   顶点数: {len(mesh.vertices)}")
                    if hasattr(mesh, 'faces'):
                        print(f"   面数: {len(mesh.faces)}")
                
                print(f"\n🎉 所有可视化已完成！")
                print(f"📁 输出目录: {output_dir.absolute()}")
                print(f"📸 可视化文件:")
                for preset in presets:
                    print(f"   - trellis_mesh_{preset}.png ({preset}视角)")
                print(f"   - trellis_generated_mesh.obj (原始mesh)")
                
            except Exception as e:
                print(f"❌ 可视化过程中出错: {e}")
                print("💡 可能原因: kiui或nvdiffrast未正确安装")
                print("💡 解决方案: pip install kiui nvdiffrast")
        else:
            print("❌ mesh解码失败")
    
    print("\n✅ TRELLIS官方模式示例 + Mesh可视化完成！")
    print("\n📚 关键点总结:")
    print("   1. 使用 TrellisStage2Pipeline() 初始化")
    print("   2. 手动调用 pipeline.cuda() 转换设备")
    print("   3. 使用 pipeline.train_mode() 和 pipeline.eval_mode() 上下文管理器")
    print("   4. Stage 1冻结，只有Stage 2(SLatFlowModel)可训练")
    print("   5. 使用扩展渲染器进行多视角mesh可视化")
    print("   6. 支持 turntable、around、corners 三种预设视角")

if __name__ == "__main__":
    main() 
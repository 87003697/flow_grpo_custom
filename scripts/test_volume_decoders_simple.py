#!/usr/bin/env python3
"""
简化版体积解码器性能测试
"""
import sys
import os
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_single_decoder(decoder_name):
    """测试单个解码器"""
    print(f"🧪 测试 {decoder_name}...")
    
    # 导入必要模块
    from hunyuan3d.pipeline import Hunyuan3DPipeline
    
    # 测试图像路径
    test_image_paths = [
        "_reference_codes/Hunyuan3D-2.1/assets/example_images/Camera_1040g34o31hmm0kqa42405np612cg9dc6aqccf38.png",
        "_reference_codes/Hunyuan3D-2.1/assets/demo.png"
    ]
    
    image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if not image_path:
        print("❌ 找不到测试图像")
        return None
    
    try:
        # 初始化pipeline
        print("🔄 初始化pipeline...")
        start_time = time.time()
        pipeline = Hunyuan3DPipeline()
        init_time = time.time() - start_time
        
        # 配置解码器
        if decoder_name == 'Hierarchical':
            try:
                from hunyuan3d.hy3dshape.models.autoencoders.volume_decoders import HierarchicalVolumeDecoding
                pipeline.pipeline.vae.volume_decoder = HierarchicalVolumeDecoding()
                print("✅ 设置分层解码器成功")
            except Exception as e:
                print(f"❌ 设置分层解码器失败: {e}")
                return None
        elif decoder_name == 'FlashVDM':
            try:
                from hunyuan3d.hy3dshape.models.autoencoders.volume_decoders import FlashVDMVolumeDecoding
                pipeline.pipeline.vae.volume_decoder = FlashVDMVolumeDecoding(topk_mode='mean')
                print("✅ 设置FlashVDM解码器成功")
            except Exception as e:
                print(f"❌ 设置FlashVDM解码器失败: {e}")
                return None
        # Vanilla不需要特殊设置
        
        # 生成mesh
        print("🎯 开始生成mesh...")
        generate_start = time.time()
        mesh = pipeline.generate_mesh(image_path)
        generate_time = time.time() - generate_start
        
        # 保存结果
        filename = f"{decoder_name.lower()}_output.glb"
        mesh.export(filename)
        
        # 获取文件大小
        file_size = os.path.getsize(filename) / (1024*1024)
        
        # 获取mesh信息
        vertex_count, face_count = get_mesh_info(filename)
        
        result = {
            'name': decoder_name,
            'init_time': init_time,
            'generate_time': generate_time,
            'total_time': init_time + generate_time,
            'file_size_mb': file_size,
            'vertex_count': vertex_count,
            'face_count': face_count,
            'filename': filename,
            'success': True
        }
        
        print(f"✅ {decoder_name} 测试成功:")
        print(f"   生成时间: {generate_time:.2f}秒")
        print(f"   文件大小: {file_size:.2f} MB")
        print(f"   顶点数: {vertex_count:,}")
        print(f"   面数: {face_count:,}")
        
        return result
        
    except Exception as e:
        print(f"❌ {decoder_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': decoder_name,
            'success': False,
            'error': str(e)
        }

def get_mesh_info(filename):
    """获取mesh信息"""
    try:
        import trimesh
        mesh_obj = trimesh.load(filename)
        
        # 处理不同类型的mesh对象
        if hasattr(mesh_obj, 'vertices'):
            # 直接是Trimesh对象
            return len(mesh_obj.vertices), len(mesh_obj.faces)
        elif hasattr(mesh_obj, 'geometry'):
            # Scene对象
            geometries = list(mesh_obj.geometry.values())
            if geometries:
                first_geom = geometries[0]
                return len(first_geom.vertices), len(first_geom.faces)
        
        return 0, 0
    except Exception as e:
        print(f"⚠️ 获取mesh信息失败: {e}")
        return 0, 0

def render_mesh(filename, decoder_name):
    """渲染mesh"""
    try:
        from hunyuan3d.hy3dshape.utils.visualizers.renderer import simple_render_mesh
        render_path = f"{decoder_name.lower()}_render.png"
        simple_render_mesh(filename, render_path)
        print(f"✅ {decoder_name} 渲染完成: {render_path}")
        return render_path
    except Exception as e:
        print(f"⚠️ {decoder_name} 渲染失败: {e}")
        return None

if __name__ == "__main__":
    print("🧪 简化版体积解码器性能测试")
    print("="*50)
    
    # 测试顺序
    decoders = ['Vanilla', 'Hierarchical', 'FlashVDM']
    results = []
    
    for decoder in decoders:
        print(f"\n{'='*30}")
        result = test_single_decoder(decoder)
        if result:
            results.append(result)
            
            # 如果成功，尝试渲染
            if result['success']:
                render_mesh(result['filename'], decoder)
    
    # 打印比较结果
    print(f"\n{'='*60}")
    print("📊 性能比较结果")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"{'解码器':<15} {'时间(秒)':<10} {'大小(MB)':<10} {'顶点数':<12} {'面数'}")
        print("-" * 60)
        
        for result in successful_results:
            print(f"{result['name']:<15} "
                  f"{result['generate_time']:<10.2f} "
                  f"{result['file_size_mb']:<10.2f} "
                  f"{result['vertex_count']:<12,} "
                  f"{result['face_count']:,}")
        
        # 找出最快的
        fastest = min(successful_results, key=lambda x: x['generate_time'])
        print(f"\n🚀 最快: {fastest['name']} ({fastest['generate_time']:.2f}秒)")
        
        # 找出质量最高的（基于顶点数）
        if any(r['vertex_count'] > 0 for r in successful_results):
            highest_quality = max(successful_results, key=lambda x: x['vertex_count'])
            print(f"🏆 最高质量: {highest_quality['name']} ({highest_quality['vertex_count']:,} 顶点)")
    else:
        print("❌ 所有测试都失败了")
    
    print("\n🎉 测试完成!") 
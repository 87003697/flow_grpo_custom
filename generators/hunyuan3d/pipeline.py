#!/usr/bin/env python3
"""
Hunyuan3D推理管道的封装
"""
import sys
import os
from pathlib import Path

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 应用必要的补丁
from patches.pytorch_rmsnorm_patch import apply_rmsnorm_patch
apply_rmsnorm_patch()

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

class Hunyuan3DPipeline:
    """Hunyuan3D推理管道的封装"""
    
    def __init__(self, model_path='tencent/Hunyuan3D-2.1'):
        self.core_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
        self.background_remover = BackgroundRemover()
        print("✅ Hunyuan3D模型加载成功")
    
    def generate_mesh(self, image_path_or_pil, output_type='trimesh', **kwargs):
        """从图像生成3D mesh"""
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGBA")
        else:
            image = image_path_or_pil
        
        # 如果是RGB图片，移除背景
        if image.mode == 'RGB':
            try:
                print("🔄 正在移除背景...")
                image = self.background_remover(image)
                print("✅ 背景移除成功")
            except Exception as e:
                print(f"⚠️ 背景移除失败: {e}")
        
        # 生成mesh - 现在传递所有额外参数
        print(f"🎯 正在生成3D mesh (格式: {output_type})...")
        
        # 如果kwargs中有num_chunks，打印出来以便调试
        if 'num_chunks' in kwargs:
            print(f"📝 使用 num_chunks = {kwargs['num_chunks']}")
        
        result = self.core_pipeline(image=image, output_type=output_type, **kwargs)
        mesh = result[0]
        print("✅ 3D mesh生成成功")
        return mesh

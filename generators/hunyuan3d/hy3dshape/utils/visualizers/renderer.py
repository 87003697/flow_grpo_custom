"""
最简化的Kiui Mesh渲染器
"""
import numpy as np
import torch
from typing import Optional

try:
    from kiui.mesh import Mesh as KiuiMesh
    from kiui.cam import OrbitCamera
    from kiui.op import safe_normalize
    import nvdiffrast.torch as dr
    KIUI_AVAILABLE = True
except ImportError as e:
    print(f'警告: kiui或nvdiffrast不可用: {e}')
    KIUI_AVAILABLE = False


class SimpleKiuiRenderer:
    """简化的Kiui渲染器"""
    
    def __init__(self, width: int = 512, height: int = 512, device: str = "cuda"):
        if not KIUI_AVAILABLE:
            raise ImportError("需要安装kiui和nvdiffrast")
        
        self.width = width
        self.height = height
        self.device = device
        
        # 初始化相机和上下文
        self.camera = OrbitCamera(width, height, fovy=50)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.background_color = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.loaded_mesh = None
    
    def load_mesh(self, mesh_path: str):
        """加载mesh文件"""
        self.loaded_mesh = KiuiMesh.load(
            str(mesh_path), 
            device=self.device, 
            resize=True, 
            bound=0.9
        )
    
    def render_single_view(self, 
                          elevation: float = 30.0,
                          azimuth: float = 45.0, 
                          distance: float = 2.0) -> np.ndarray:
        """渲染单个视图"""
        # 设置相机
        self.camera.from_angle(elevation=elevation, azimuth=azimuth, is_degree=True)
        self.camera.radius = distance
        self.camera.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # 获取相机矩阵
        pose = torch.from_numpy(self.camera.pose.astype(np.float32)).to(self.device)
        proj = torch.from_numpy(self.camera.perspective.astype(np.float32)).to(self.device)
        
        # 变换顶点
        v_cam = torch.matmul(
            torch.nn.functional.pad(self.loaded_mesh.v, pad=(0, 1), mode='constant', value=1.0), 
            torch.inverse(pose).T
        ).float().unsqueeze(0)
        v_clip = v_cam @ proj.T
        
        # 光栅化
        rast, _ = dr.rasterize(self.glctx, v_clip, self.loaded_mesh.f, (self.height, self.width))
        alpha = (rast[..., 3:] > 0).float()
        
        # 简单的lambertian渲染
        if self.loaded_mesh.vc is not None:
            # 使用顶点颜色
            albedo, _ = dr.interpolate(self.loaded_mesh.vc.unsqueeze(0).contiguous(), rast, self.loaded_mesh.f)
        else:
            # 默认灰色
            albedo = torch.ones_like(v_cam[..., :3]) * 0.7
            albedo, _ = dr.interpolate(albedo, rast, self.loaded_mesh.f)
        
        # 简单光照
        if hasattr(self.loaded_mesh, 'vn') and self.loaded_mesh.vn is not None:
            normal, _ = dr.interpolate(self.loaded_mesh.vn.unsqueeze(0).contiguous(), rast, self.loaded_mesh.fn)
            normal = safe_normalize(normal)
            
            light_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            lambertian = 0.5 + 0.5 * (normal @ light_dir).float().clamp(min=0)
            albedo = albedo * lambertian.unsqueeze(-1)
        
        # 合成最终图像
        image = albedo * alpha + self.background_color * (1 - alpha)
        
        # 转换为numpy
        buffer = image[0].detach().cpu().numpy()
        buffer = np.clip(buffer * 255, 0, 255).astype(np.uint8)
        
        return buffer


def simple_render_mesh(mesh_path: str, save_path: str, device: str = "cuda") -> str:
    """简单的mesh渲染函数"""
    # 创建渲染器并加载mesh
    renderer = SimpleKiuiRenderer(device=device)
    renderer.load_mesh(mesh_path)
    
    # 渲染图像
    image = renderer.render_single_view(elevation=30, azimuth=45, distance=2.0)
    
    # 保存图像
    from PIL import Image
    img = Image.fromarray(image)
    img.save(save_path)
    
    print(f"💾 渲染已保存: {save_path}")
    return save_path


def render_mesh_for_training(mesh_path: str, output_path: str, device: str = "cuda") -> str:
    """训练时渲染mesh"""
    return simple_render_mesh(mesh_path, output_path, device)


if __name__ == "__main__":
    # 测试
    import trimesh
    
    # 创建测试mesh并保存
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.export("test_mesh.obj")
    
    # 渲染测试
    result = simple_render_mesh("test_mesh.obj", "test_render.png")
    print(f"✅ 测试完成: {result}")

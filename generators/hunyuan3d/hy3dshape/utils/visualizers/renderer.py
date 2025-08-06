"""
最简化的Kiui Mesh渲染器
"""
import numpy as np
import torch
from typing import Optional, List, Tuple
from PIL import Image
import os

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
    
    def load_mesh_from_trimesh(self, mesh):
        """从trimesh对象加载mesh"""
        # 将trimesh转换为kiui mesh
        # 确保mesh.vertices是numpy数组
        if hasattr(mesh.vertices, 'cpu'):
            vertices_np = mesh.vertices.cpu().numpy().astype(np.float32)
        else:
            vertices_np = np.array(mesh.vertices, dtype=np.float32)
        
        if hasattr(mesh.faces, 'cpu'):
            faces_np = mesh.faces.cpu().numpy().astype(np.int32)
        else:
            faces_np = np.array(mesh.faces, dtype=np.int32)
            
        vertices = torch.from_numpy(vertices_np).to(self.device)
        faces = torch.from_numpy(faces_np).to(self.device)
        
        # 创建kiui mesh对象
        self.loaded_mesh = KiuiMesh(device=self.device)
        self.loaded_mesh.v = vertices
        self.loaded_mesh.f = faces
        
        # 计算顶点法向量
        self.loaded_mesh.auto_normal()
        
        # 重新缩放和居中
        self.loaded_mesh.auto_size()
    
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
    
    def render_multiple_views(self, 
                            views: List[Tuple[float, float, float]] = None,
                            preset: str = "turntable") -> List[np.ndarray]:
        """渲染多个视图
        
        Args:
            views: 视图列表，每个元素为(elevation, azimuth, distance)
            preset: 预设视角模式，可选"turntable"、"around"、"corners"
            
        Returns:
            渲染图像列表
        """
        if views is None:
            if preset == "turntable":
                # 水平环绕视图 (固定elevation=30°)
                views = [(30.0, azimuth, 2.0) for azimuth in range(0, 360, 45)]
            elif preset == "around":
                # 3x3网格视图
                elevations = [15.0, 30.0, 45.0]
                azimuths = [0.0, 120.0, 240.0]
                views = [(elev, azim, 2.0) for elev in elevations for azim in azimuths]
            elif preset == "corners":
                # 4个角度视图
                views = [
                    (30.0, 45.0, 2.0),   # 右前
                    (30.0, 135.0, 2.0),  # 右后  
                    (30.0, 225.0, 2.0),  # 左后
                    (30.0, 315.0, 2.0),  # 左前
                ]
            else:
                raise ValueError(f"未知预设: {preset}")
        
        rendered_images = []
        for elevation, azimuth, distance in views:
            image = self.render_single_view(elevation, azimuth, distance)
            rendered_images.append(image)
        
        return rendered_images


def create_grid_image(images: List[np.ndarray], 
                     grid_size: Tuple[int, int] = None,
                     padding: int = 10,
                     background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """将多个图像组合成网格
    
    Args:
        images: 图像列表
        grid_size: 网格大小(rows, cols)，如果为None则自动计算
        padding: 图像间间距
        background_color: 背景颜色
        
    Returns:
        组合后的网格图像
    """
    if not images:
        raise ValueError("图像列表为空")
    
    # 获取单个图像尺寸
    img_h, img_w = images[0].shape[:2]
    
    # 自动计算网格大小
    if grid_size is None:
        num_images = len(images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # 计算总尺寸
    total_w = cols * img_w + (cols - 1) * padding
    total_h = rows * img_h + (rows - 1) * padding
    
    # 创建背景
    if len(images[0].shape) == 3:
        grid_image = np.full((total_h, total_w, 3), background_color, dtype=np.uint8)
    else:
        grid_image = np.full((total_h, total_w), background_color[0], dtype=np.uint8)
    
    # 放置图像
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        
        y_start = row * (img_h + padding)
        x_start = col * (img_w + padding)
        y_end = y_start + img_h
        x_end = x_start + img_w
        
        grid_image[y_start:y_end, x_start:x_end] = img
    
    return grid_image


def simple_render_mesh(mesh_path: str, save_path: str, device: str = "cuda") -> str:
    """简单的mesh渲染函数"""
    # 创建渲染器并加载mesh
    renderer = SimpleKiuiRenderer(device=device)
    renderer.load_mesh(mesh_path)
    
    # 渲染图像
    image = renderer.render_single_view(elevation=30, azimuth=45, distance=2.0)
    
    # 保存图像
    img = Image.fromarray(image)
    img.save(save_path)
    
    print(f"💾 渲染已保存: {save_path}")
    return save_path


def render_mesh_multiple_views(mesh_path: str = None, 
                              mesh_trimesh = None,
                              save_path: str = "mesh_views.png",
                              preset: str = "turntable",
                              device: str = "cuda") -> str:
    """渲染mesh的多个视角并保存为网格图像
    
    Args:
        mesh_path: mesh文件路径 (与mesh_trimesh二选一)
        mesh_trimesh: trimesh对象 (与mesh_path二选一)
        save_path: 保存路径
        preset: 视角预设，可选"turntable"、"around"、"corners"
        device: 设备
        
    Returns:
        保存的文件路径
    """
    # 创建渲染器
    renderer = SimpleKiuiRenderer(device=device)
    
    # 加载mesh
    if mesh_path is not None:
        renderer.load_mesh(mesh_path)
    elif mesh_trimesh is not None:
        renderer.load_mesh_from_trimesh(mesh_trimesh)
    else:
        raise ValueError("必须提供mesh_path或mesh_trimesh")
    
    # 渲染多个视图
    images = renderer.render_multiple_views(preset=preset)
    
    # 创建网格图像
    if preset == "turntable":
        grid_size = (2, 4)  # 8个视图，2x4网格
    elif preset == "around":
        grid_size = (3, 3)  # 9个视图，3x3网格
    elif preset == "corners":
        grid_size = (2, 2)  # 4个视图，2x2网格
    else:
        grid_size = None  # 自动计算
    
    grid_image = create_grid_image(images, grid_size=grid_size)
    
    # 保存网格图像
    img = Image.fromarray(grid_image)
    img.save(save_path)
    
    print(f"🎨 多视角渲染已保存: {save_path}")
    print(f"   预设: {preset}, 视图数量: {len(images)}")
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
    
    # 测试单视图渲染
    result = simple_render_mesh("test_mesh.obj", "test_render.png")
    print(f"✅ 单视图测试完成: {result}")
    
    # 测试多视图渲染
    result = render_mesh_multiple_views(mesh_path="test_mesh.obj", 
                                       save_path="test_multiview.png",
                                       preset="turntable")
    print(f"✅ 多视图测试完成: {result}")

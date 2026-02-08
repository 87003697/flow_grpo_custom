"""
edit4shape 渲染器模块

所有渲染器都继承自 BaseRenderer，遵循统一的 7 阶段渲染流水线:
    Stage 1: Input Preparation (输入准备)
    Stage 2: Camera Transform (相机变换)
    Stage 3: Geometry Processing (几何处理)
    Stage 4: Rasterization Core (光栅化/渲染核心)
    Stage 5: Attribute Interpolation (属性插值/采样)
    Stage 6: Post-processing (后处理)
    Stage 7: Output Assembly (输出组装)

渲染器列表:
    - BaseRenderer: 抽象基类
    - TrellisMeshRasterizer: Mesh 光栅化器 (nvdiffrast)
    - GaussianRenderer: 3D Gaussian Splatting 渲染器 (需要 trellis 模块)
    - VoxelRenderer: 基础体素渲染器 (o_voxel)
    - PbrVoxelRenderer: PBR 着色体素渲染器
    - DiffVoxelRenderer: 可微体素渲染器 (STE)
    - SoftVoxelRenderer: 纯 PyTorch 可微体素渲染器
    - Quad12NormalRenderer: 12-Quad 法线渲染器
"""

# 基类和工具
from edit4shape.renderers.base_renderer import (
    # 数据结构
    RenderConfig,
    CameraData,
    RasterOutput,
    RenderOutput,
    # 工具函数
    intrinsics_to_projection,
    depth_to_normal,
    camera_normal_to_vis,
    ssaa_downsample,
    # 基类
    BaseRenderer,
    PBRPostProcessMixin,
)

# Mesh 渲染器
from edit4shape.renderers.sparseflex_trellis import (
    TrellisMeshRasterizer,
    MeshRasterData,
)

# Gaussian 渲染器 - 条件导入（需要 trellis 模块）
try:
    from edit4shape.renderers.gaussian_splatting_trellis import (
        GaussianRenderer,
        GaussianPipeConfig,
        GaussianCameraData,
        GaussianRasterResult,
        gaussian_rasterize,
    )
    _HAS_GAUSSIAN = True
except ImportError:
    _HAS_GAUSSIAN = False
    GaussianRenderer = None
    GaussianPipeConfig = None
    GaussianCameraData = None
    GaussianRasterResult = None
    gaussian_rasterize = None

__all__ = [
    # 基类和工具
    'RenderConfig',
    'CameraData',
    'RasterOutput',
    'RenderOutput',
    'intrinsics_to_projection',
    'depth_to_normal',
    'camera_normal_to_vis',
    'ssaa_downsample',
    'BaseRenderer',
    'PBRPostProcessMixin',
    # Mesh
    'TrellisMeshRasterizer',
    'MeshRasterData',
    # Gaussian (可选)
    'GaussianRenderer',
    'GaussianPipeConfig',
    'GaussianCameraData',
    'GaussianRasterResult',
    'gaussian_rasterize',
    # Voxel / Soft Voxel / Quad12 渲染器暂不在本地 trellis 流程中使用
]

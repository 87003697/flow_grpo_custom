"""
edit4shape 渲染器模块

渲染器列表:
    - BaseRenderer: 抽象基类
    - TrellisMeshRasterizer: Mesh 光栅化器 (nvdiffrast)
    - GaussianRenderer: 3D Gaussian Splatting 渲染器 (需要 trellis 模块)
    - VoxelRenderer: 基础体素渲染器 (o_voxel)
    - PbrVoxelRenderer: PBR 着色体素渲染器
    - DiffVoxelRenderer: 可微体素渲染器 (STE)
    - Hybrid26NormalRenderer: 26-neighbor 混合法线渲染器 (subs 可微)
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

# Voxel 渲染器
from edit4shape.renderers.ovoxel_trellis2 import (
    VoxelRenderer,
    PbrVoxelRenderer,
    DiffVoxelRenderer,
    VoxelProxy,
    VoxelRasterData,
    load_envmap,
)

# Hybrid26 法线渲染器
from edit4shape.renderers.hybrid_normal_renderer import (
    Hybrid26NormalRenderer,
)


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
    # Voxel
    'VoxelRenderer',
    'PbrVoxelRenderer',
    'DiffVoxelRenderer',
    'VoxelProxy',
    'VoxelRasterData',
    'load_envmap',
    # Hybrid26
    'Hybrid26NormalRenderer',
]

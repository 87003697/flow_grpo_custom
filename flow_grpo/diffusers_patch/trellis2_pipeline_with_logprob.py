import os
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor
from trellis2.representations import MeshWithVoxel
from trellis2.renderers import PbrMeshRenderer, EnvMap
from trellis2.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames,
    render_snapshot,
    render_multiview,
    make_pbr_vis_frames,
)

# 默认 HDRI 环境贴图路径
_TRELLIS2_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "_reference_codes", "TRELLIS.2")
_DEFAULT_HDRI_PATH = os.path.join(_TRELLIS2_ROOT, "assets", "hdri", "forest.exr")


def _load_exr_image(path: str) -> np.ndarray:
    """加载 EXR 格式的 HDR 图像。"""
    # 仅使用 OpenEXR 原生读取，避免回落到其他解码器
    import OpenEXR, Imath
    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []
    for c in ("R", "G", "B"):
        buf = exr.channel(c, pt)
        arr = np.frombuffer(buf, dtype=np.float32)
        channels.append(arr)
    img = np.stack(channels, axis=-1).reshape(h, w, 3)
    return img


class Trellis2PipelineWithLogProb(Trellis2ImageTo3DPipeline):
    """Trellis2 推理管线，补充分段采样接口与模型暴露，供 NFT 训练/简化采样使用。"""

    @staticmethod
    def from_pretrained(path: str, dino_local_path: Optional[str] = None):
        base = super(Trellis2PipelineWithLogProb, Trellis2PipelineWithLogProb).from_pretrained(path, dino_local_path)
        new = Trellis2PipelineWithLogProb()
        new.__dict__ = base.__dict__
        return new

    def get_trainable_models(self) -> Dict[str, torch.nn.Module]:
        """返回所有可训练模块字典。"""
        return self.models

    def prepare_image_conditions(self, images, resolution: int = 1024):
        """包装 get_cond，只返回 cond。"""
        cond_dict = self.get_cond(images, resolution=resolution, include_neg_cond=False)
        return cond_dict.get("cond")

    @torch.no_grad()
    def stage_1(self, cond: Dict, ss_resolution: int, num_samples: int = 1, params: Dict = {}):
        """稀疏结构采样，返回 (coords, sampler_out)。"""
        flow = self.models["sparse_structure_flow_model"]
        # 注意：使用 .to(device) 而非 device= 参数，以匹配官方实现的随机数生成行为（CPU 生成后移动到 GPU）
        noise = torch.randn(
            num_samples, flow.in_channels, flow.resolution, flow.resolution, flow.resolution
        ).to(self.device)  # 形状: (B, C, R, R, R)
        out = self.sparse_structure_sampler.sample(
            flow, noise, **cond, **{**self.sparse_structure_sampler_params, **params}, verbose=True
        )
        decoded = self.models["sparse_structure_decoder"](out.samples) > 0  # 形状: (B, 1, R, R, R)
        if ss_resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // ss_resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()  # 形状: (N, 4)
        # 验证坐标批次一致性
        batch_ids = coords[:, 0].unique()
        expected = num_samples if "cond" not in cond else cond["cond"].shape[0]
        if batch_ids.numel() != expected:
            raise RuntimeError(
                f"stage_1 coords batch mismatch: got {batch_ids.tolist()} (count={batch_ids.numel()}) "
                f"but expected {int(expected)}; decoded.shape={tuple(decoded.shape)}"
            )
        if coords.shape[0] == 0:
            raise RuntimeError(
                f"stage_1 生成空坐标: decoded.shape={tuple(decoded.shape)}, num_samples={num_samples}"
            )
        return coords, out

    @torch.no_grad()
    def stage_2_shape(self, cond: Dict, coords: torch.Tensor, resolution: int = 1024, params: Dict = {}):
        """形状 SLat 采样，返回 (slat, sampler_out)。"""
        flow = self.models[f"shape_slat_flow_model_{resolution}"]
        # 注意：使用 .to(device) 以匹配官方随机数生成行为
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow.in_channels).to(self.device),  # 形状: (N, C)
            coords=coords,
        )
        out = self.shape_slat_sampler.sample(
            flow, noise, **cond, **{**self.shape_slat_sampler_params, **params}, verbose=True
        )
        std = torch.tensor(self.shape_slat_normalization["std"], device=self.device)  # 形状: (C,)
        mean = torch.tensor(self.shape_slat_normalization["mean"], device=self.device)  # 形状: (C,)
        slat = out.samples * std + mean  # 形状: SparseTensor(N, C)
        return slat, out

    @torch.no_grad()
    def stage_2_shape_cascade(
        self,
        lr_cond: Dict,
        hr_cond: Dict,
        coords: torch.Tensor,
        lr_resolution: int = 512,
        hr_resolution: int = 1024,
        max_num_tokens: int = 49152,
        params: Dict = {},
    ) -> Tuple[SparseTensor, SparseTensor, int]:
        """
        级联 Shape SLat 采样：
        1. 先用 512 模型生成粗略 shape latent
        2. 通过 decoder.upsample 获取高分辨率坐标
        3. 用 1024 模型在高分辨率坐标上精细采样
        """
        # Step 1: 低分辨率采样 (512)
        # ========================================
        flow_lr = self.models.get(f"shape_slat_flow_model_{lr_resolution}")
        if flow_lr is None:
            raise KeyError(f"Cascade 需要 shape_slat_flow_model_{lr_resolution}")
        
        coords = coords.to(self.device)  # 确保输入坐标在正确设备上
        # 注意：使用 .to(device) 以匹配官方随机数生成行为
        noise_lr = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_lr.in_channels).to(self.device),
            coords=coords,
        )
        out_lr = self.shape_slat_sampler.sample(
            flow_lr, noise_lr, **lr_cond, **{**self.shape_slat_sampler_params, **params}, 
            verbose=True, tqdm_desc="Sampling shape SLat (LR)"
        )
        
        # 反规范化
        std = torch.tensor(self.shape_slat_normalization["std"], device=self.device)
        mean = torch.tensor(self.shape_slat_normalization["mean"], device=self.device)
        slat_lr = out_lr.samples * std + mean

        # Step 2: 坐标上采样
        decoder = self.models["shape_slat_decoder"]
        hr_coords = decoder.upsample(slat_lr, upsample_times=4)
        
        # 动态调整分辨率以控制 token 数量
        actual_hr_resolution = hr_resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (actual_hr_resolution // 16)).int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0).to(self.device)  # 确保在正确设备上
            num_tokens = hr_coords_unique.shape[0]
            
            if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
                if actual_hr_resolution != hr_resolution:
                    print(f"[Cascade] Token 数超限 ({num_tokens} >= {max_num_tokens})，分辨率降至 {actual_hr_resolution}")
                break
            actual_hr_resolution -= 128

        # Step 3: 高分辨率采样 (1024)
        flow_hr = self.models.get(f"shape_slat_flow_model_{actual_hr_resolution}") or self.models.get("shape_slat_flow_model_1024")
        if flow_hr is None:
            raise KeyError(f"Cascade 需要 shape_slat_flow_model_1024 或 shape_slat_flow_model_{actual_hr_resolution}")
            
        # 注意：使用 .to(device) 以匹配官方随机数生成行为
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], flow_hr.in_channels).to(self.device),
            coords=hr_coords_unique,
        )
        out_hr = self.shape_slat_sampler.sample(
            flow_hr, noise_hr, **hr_cond, **{**self.shape_slat_sampler_params, **params}, 
            verbose=True, tqdm_desc="Sampling shape SLat (HR)"
        )
        slat_hr = out_hr.samples * std + mean

        return slat_hr, slat_lr, actual_hr_resolution

    @torch.no_grad()
    def stage_2_tex(self, cond: Dict, shape_slat: SparseTensor, params: Dict = {}):
        """纹理 SLat 采样，返回 (tex_slat, sampler_out)。"""
        std_s = torch.tensor(self.shape_slat_normalization["std"], device=self.device)  # 形状: (C_shape,)
        mean_s = torch.tensor(self.shape_slat_normalization["mean"], device=self.device)  # 形状: (C_shape,)
        shape_norm = (shape_slat - mean_s) / std_s  # 形状: SparseTensor(N, C_shape)

        flow = self.models["tex_slat_flow_model_1024"]
        in_ch = flow.in_channels if hasattr(flow, "in_channels") else flow[0].in_channels  # 形状: 标量
        # 注意：使用 .to(device) 以匹配官方随机数生成行为
        noise = shape_norm.replace(
            feats=torch.randn(shape_norm.coords.shape[0], in_ch - shape_norm.feats.shape[1]).to(self.device)
        )  # 形状: SparseTensor(N, C_tex_noise)
        out = self.tex_slat_sampler.sample(
            flow, noise, concat_cond=shape_norm, **cond, **{**self.tex_slat_sampler_params, **params}, verbose=True
        )
        std_t = torch.tensor(self.tex_slat_normalization["std"], device=self.device)  # 形状: (C_tex,)
        mean_t = torch.tensor(self.tex_slat_normalization["mean"], device=self.device)  # 形状: (C_tex,)
        tex = out.samples * std_t + mean_t  # 形状: SparseTensor(N, C_tex)
        return tex, out

    def get_flow_module(self, kind: str, resolution: Optional[int] = None, unwrap_ddp: bool = True):
        """
        统一获取 flow 模块，支持分辨率回退并可解包 DDP/FSDP wrapper。
        
        Args:
            kind: "structure" | "shape_slat" | "tex_slat"
            resolution: 需要的分辨率（可选，用于选择特定分支）
            unwrap_ddp: 是否自动返回 .module（多 GPU 训练下常见）
        """
        if kind == "structure":
            candidates = [self.models.get("sparse_structure_flow_model")]
        elif kind == "shape_slat":
            candidates = [
                self.models.get(f"shape_slat_flow_model_{resolution}") if resolution else None,
                self.models.get("shape_slat_flow_model"),
                self.models.get("shape_slat_flow_model_1024"),
                self.models.get("shape_slat_flow_model_512"),
            ]
        elif kind == "tex_slat":
            candidates = [
                self.models.get(f"tex_slat_flow_model_{resolution}") if resolution else None,
                self.models.get("tex_slat_flow_model_1024"),
                self.models.get("tex_slat_flow_model"),
            ]
        else:
            raise ValueError(f"unknown flow kind: {kind}")

        module = next((m for m in candidates if m is not None), None)
        if module is None:
            raise KeyError(f"flow module not found for kind={kind}, resolution={resolution}")

        return module.module if (unwrap_ddp and hasattr(module, "module")) else module

    def set_shape_flow_model(self, model: "torch.nn.Module", resolution: int = 1024) -> None:
        """将 LoRA 训练后的形状模型传回 pipeline。
        
        Args:
            model: 训练后的模型
            resolution: 模型对应的分辨率（512 或 1024）
        """
        key = f"shape_slat_flow_model_{resolution}"
        if key in self.models:
            self.models[key] = model
        else:
            # fallback：尝试设置通用 key
            for fallback_key in ["shape_slat_flow_model_1024", "shape_slat_flow_model", "shape_slat_flow_model_512"]:
                if fallback_key in self.models:
                    self.models[fallback_key] = model
                    return
            self.models[key] = model
    
    def get_shape_flow_model(self, resolution: int = 1024) -> "torch.nn.Module":
        """获取指定分辨率的形状 flow 模型。"""
        key = f"shape_slat_flow_model_{resolution}"
        if key in self.models:
            return self.models[key]
        # fallback
        for fallback_key in ["shape_slat_flow_model_1024", "shape_slat_flow_model"]:
            if fallback_key in self.models:
                return self.models[fallback_key]
        raise KeyError(f"未找到形状模型 {resolution}，已有 keys={list(self.models.keys())}")

    def get_tex_flow_model(self) -> "torch.nn.Module":
        """获取纹理 flow 模型。"""
        for key in ["tex_slat_flow_model_1024", "tex_slat_flow_model"]:
            if key in self.models:
                return self.models[key]
        raise KeyError(f"未找到纹理模型，已有 keys={list(self.models.keys())}")

    def set_tex_flow_model(self, model: "torch.nn.Module") -> None:
        """将 LoRA 训练后的纹理模型传回 pipeline。"""
        for key in ["tex_slat_flow_model_1024", "tex_slat_flow_model"]:
            if key in self.models:
                self.models[key] = model
                return
        self.models["tex_slat_flow_model_1024"] = model

    @torch.no_grad()
    def export_mesh(self, shape_slat: SparseTensor, tex_slat: Optional[SparseTensor] = None, resolution: int = 1024):
        """
        将 shape_slat/tex_slat 解码为 MeshWithVoxel；若 tex_slat 为空则输出无纹理 mesh。
        
        纹理解码依赖形状解码产生的 subdivision 信息 (subs)，需配对使用。
        """
        # 设置 decoder 分辨率
        self.models["shape_slat_decoder"].set_resolution(resolution)
        
        # 形状解码：return_subs=True 返回 (meshes, subs)
        meshes, subs = self.models["shape_slat_decoder"](shape_slat, return_subs=True)
        mesh_obj = meshes[0] if isinstance(meshes, list) else meshes
        
        # 纹理解码（需要 guide_subs）
        tex_dec = None
        if tex_slat is not None and ("tex_slat_decoder" in self.models):
            tex_dec = self.models["tex_slat_decoder"](tex_slat, guide_subs=subs) * 0.5 + 0.5
            if isinstance(tex_dec, list):
                tex_dec = tex_dec[0]

        mesh = MeshWithVoxel(
            mesh_obj.vertices,
            mesh_obj.faces,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1.0 / float(resolution),
            coords=(tex_dec.coords[:, 1:] if tex_dec is not None else None),
            attrs=(tex_dec.feats if tex_dec is not None else None),
            voxel_shape=(
                torch.Size([*tex_dec.shape, *tex_dec.spatial_shape])
                if tex_dec is not None else None
            ),
            layout=self.pbr_attr_layout,
        )
        return mesh

    # =========================================================================
    # PBR 渲染支持
    # =========================================================================

    def get_default_envmap(self) -> EnvMap:
        """加载默认 HDRI 环境贴图。"""
        img = _load_exr_image(_DEFAULT_HDRI_PATH)
        return EnvMap(torch.tensor(img, dtype=torch.float32, device=self.device))

    def render_pbr_snapshot(
        self,
        mesh: MeshWithVoxel,
        envmap: Optional[EnvMap] = None,
        resolution: int = 512,
        nviews: int = 4,
    ) -> Dict[str, List[np.ndarray]]:
        """渲染 mesh 的 PBR 快照（固定视角）。

        Args:
            mesh: MeshWithVoxel 对象
            envmap: 环境贴图，默认使用 forest.exr
            resolution: 渲染分辨率
            nviews: 视角数量

        Returns:
            包含 'shaded', 'normal', 'base_color' 等通道的帧列表
        """
        envmap = envmap or self.get_default_envmap()
        return render_snapshot(mesh, resolution=resolution, nviews=nviews, envmap=envmap)

    def render_pbr_multiview(
        self,
        mesh: MeshWithVoxel,
        envmap: Optional[EnvMap] = None,
        resolution: int = 512,
        nviews: int = 30,
    ) -> Tuple[List[np.ndarray], List[torch.Tensor], List[torch.Tensor]]:
        """渲染 mesh 的 PBR 多视角图像（Hammersley 采样）。

        Returns:
            (color_frames, extrinsics, intrinsics)
        """
        envmap = envmap or self.get_default_envmap()
        return render_multiview(mesh, resolution=resolution, nviews=nviews, envmap=envmap)

    def render_pbr_frames(
        self,
        mesh: MeshWithVoxel,
        yaws: List[float],
        pitchs: List[float],
        envmap: Optional[EnvMap] = None,
        resolution: int = 512,
        r: float = 2.0,
        fov: float = 40.0,
    ) -> Dict[str, List[np.ndarray]]:
        """按指定相机参数渲染 PBR 帧序列。

        Args:
            mesh: MeshWithVoxel 对象
            yaws: 方位角列表（弧度）
            pitchs: 俯仰角列表（弧度）
            envmap: 环境贴图
            resolution: 渲染分辨率
            r: 相机距离
            fov: 视场角（度）

        Returns:
            包含各渲染通道的帧列表
        """
        envmap = envmap or self.get_default_envmap()
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
        return render_frames(mesh, extrinsics, intrinsics, {'resolution': resolution}, envmap=envmap)

    @staticmethod
    def make_pbr_vis_panel(render_result: Dict[str, List[np.ndarray]], resolution: int = 512) -> List[np.ndarray]:
        """将 PBR 渲染结果拼成可视化面板（shaded + normal + base_color 等）。"""
        return make_pbr_vis_frames(render_result, resolution=resolution)


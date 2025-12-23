import torch
from typing import Dict, Tuple, Optional
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor
from trellis2.representations import MeshWithVoxel


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

    def prepare_image_conditions(self, images, resolution: int = 1024, include_neg_cond: bool = True):
        """包装 get_cond，返回 (cond, neg_cond)。"""
        cond_dict = self.get_cond(images, resolution=resolution, include_neg_cond=include_neg_cond)
        cond = cond_dict.get("cond")
        neg = cond_dict.get("neg_cond") if include_neg_cond else None
        return cond, neg

    # # === 简化分阶段采样接口（无 logprob/时间序列） ===
    # @torch.no_grad()
    # def stage_1(self, cond: Dict, ss_resolution: int, num_samples: int = 1, params: Dict = {}):
    #     """稀疏结构采样（同 stage_sparse，命名兼容旧调用）。"""
    #     return self.stage_sparse(cond=cond, ss_resolution=ss_resolution, num_samples=num_samples, params=params)

    # @torch.no_grad()
    # def stage_2_shape(self, cond: Dict, coords: torch.Tensor, resolution: int = 1024, params: Dict = {}):
    #     """形状 SLat 采样（同 stage_shape，命名兼容旧调用）。"""
    #     return self.stage_shape(cond=cond, coords=coords, resolution=resolution, params=params)

    # @torch.no_grad()
    # def stage_2_tex(self, cond: Dict, shape_slat: SparseTensor, params: Dict = {}):
    #     """纹理 SLat 采样（同 stage_tex，命名兼容旧调用）。"""
    #     return self.stage_tex(cond=cond, shape_slat=shape_slat, params=params)

    @torch.no_grad()
    def stage_1(self, cond: Dict, ss_resolution: int, num_samples: int = 1, params: Dict = {}):
        """稀疏结构采样，返回 (coords, sampler_out)。"""
        flow = self.models["sparse_structure_flow_model"]
        noise = torch.randn(
            num_samples, flow.in_channels, flow.resolution, flow.resolution, flow.resolution,
            device=self.device
        )  # 形状: (B, C, R, R, R)
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
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow.in_channels, device=self.device),  # 形状: (N, C)
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
    def stage_2_tex(self, cond: Dict, shape_slat: SparseTensor, params: Dict = {}):
        """纹理 SLat 采样，返回 (tex_slat, sampler_out)。"""
        std_s = torch.tensor(self.shape_slat_normalization["std"], device=self.device)  # 形状: (C_shape,)
        mean_s = torch.tensor(self.shape_slat_normalization["mean"], device=self.device)  # 形状: (C_shape,)
        shape_norm = (shape_slat - mean_s) / std_s  # 形状: SparseTensor(N, C_shape)

        flow = self.models["tex_slat_flow_model_1024"]
        in_ch = flow.in_channels if hasattr(flow, "in_channels") else flow[0].in_channels  # 形状: 标量
        noise = shape_norm.replace(
            feats=torch.randn(shape_norm.coords.shape[0], in_ch - shape_norm.feats.shape[1], device=self.device)
        )  # 形状: SparseTensor(N, C_tex_noise)
        out = self.tex_slat_sampler.sample(
            flow, noise, concat_cond=shape_norm, **cond, **{**self.tex_slat_sampler_params, **params}, verbose=True
        )
        std_t = torch.tensor(self.tex_slat_normalization["std"], device=self.device)  # 形状: (C_tex,)
        mean_t = torch.tensor(self.tex_slat_normalization["mean"], device=self.device)  # 形状: (C_tex,)
        tex = out.samples * std_t + mean_t  # 形状: SparseTensor(N, C_tex)
        return tex, out

    # ===== 兼容接口：评估/训练侧调用 =====
    def _resolve_structure_flow_module(self):
        return self.models.get("sparse_structure_flow_model")

    def _resolve_slat_flow_module(self):
        return (
            self.models.get("shape_slat_flow_model_1024")
            or self.models.get("shape_slat_flow_model")
            or self.models.get("shape_slat_flow_model_512")
        )

    @torch.no_grad()
    def export_mesh(self, shape_slat: SparseTensor, tex_slat: Optional[SparseTensor] = None, resolution: int = 1024):
        """
        将 shape_slat/tex_slat 解码为 MeshWithVoxel；若 tex_slat 为空则输出无纹理 mesh。
        """
        # 形状解码
        shape_dec = self.models["shape_slat_decoder"](shape_slat)
        # 纹理解码（可选）
        tex_dec = None
        if tex_slat is not None and ("tex_slat_decoder" in self.models):
            tex_dec = self.models["tex_slat_decoder"](tex_slat)

        mesh_obj = shape_dec
        if hasattr(mesh_obj, "fill_holes"):
            mesh_obj.fill_holes()

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


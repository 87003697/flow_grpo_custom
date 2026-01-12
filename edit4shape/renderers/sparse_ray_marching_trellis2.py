import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SparseLevel:
    """单层稀疏体积数据"""
    coords: torch.Tensor      # (N, 3) int
    density: torch.Tensor     # (N,) 或 (N, 8) float
    resolution: int           # 该层分辨率
    hash_table: torch.Tensor  # 哈希表


class MultiResolutionSparseVolume:
    """
    多分辨率稀疏体积。
    
    支持从 subs 构建层次结构，并进行高效的层次化 Ray Marching。
    """
    
    def __init__(
        self,
        aabb: Tuple[float, float] = (-0.5, 0.5),
        device: str = 'cuda',
    ):
        self.aabb_min = aabb[0]
        self.aabb_max = aabb[1]
        self.device = device
        self.levels: List[SparseLevel] = []
    
    def build_from_subs(
        self,
        subs: List,  # List[SparseTensor]，从粗到细
        base_resolution: int = 64,
    ):
        """
        从 subs 构建多分辨率稀疏体积。
        
        Args:
            subs: 每层的 SparseTensor，sub.coords (N, 4), sub.feats (N, 8)
            base_resolution: 第 0 层的父分辨率
        """
        self.levels = []
        
        for i, sub in enumerate(subs):
            parent_res = base_resolution * (2 ** i)
            child_res = parent_res * 2  # 展开后的分辨率
            
            # 提取坐标
            coords = sub.coords[:, 1:] if sub.coords.shape[1] == 4 else sub.coords
            coords = coords.to(self.device)
            
            # 特征 -> 密度
            if sub.feats.shape[-1] == 8:
                # 展开 8 个子 voxel
                density = torch.sigmoid(sub.feats)  # (N, 8)
                coords_expanded, density_flat = self._expand_subdivision(
                    coords, density, parent_res
                )
            else:
                coords_expanded = coords
                density_flat = torch.sigmoid(sub.feats.squeeze(-1))
            
            # 构建哈希表
            hash_table = self._build_hash_table(coords_expanded, density_flat, child_res)
            
            level = SparseLevel(
                coords=coords_expanded,
                density=density_flat,
                resolution=child_res,
                hash_table=hash_table,
            )
            self.levels.append(level)
            
            print(f"Level {i}: res={child_res}, voxels={coords_expanded.shape[0]}")
    
    def _expand_subdivision(
        self,
        coords: torch.Tensor,    # (N, 3)
        density: torch.Tensor,   # (N, 8)
        parent_res: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """展开 8 个子 voxel"""
        N = coords.shape[0]
        
        offsets = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], device=self.device)  # (8, 3)
        
        # (N, 1, 3) * 2 + (1, 8, 3) -> (N, 8, 3)
        coords_expanded = coords.unsqueeze(1) * 2 + offsets.unsqueeze(0)
        coords_flat = coords_expanded.reshape(-1, 3)  # (N*8, 3)
        density_flat = density.reshape(-1)            # (N*8,)
        
        return coords_flat, density_flat
    
    def _build_hash_table(
        self,
        coords: torch.Tensor,
        density: torch.Tensor,
        resolution: int,
    ) -> torch.Tensor:
        """构建哈希表"""
        N = coords.shape[0]
        
        # 确保 density 是 float32
        density = density.float()
        
        # 线性索引
        keys = (coords[:, 0] * resolution * resolution +
                coords[:, 1] * resolution +
                coords[:, 2]).long()
        
        # 过滤越界
        valid = (keys >= 0) & (keys < resolution ** 3)
        keys = keys[valid]
        density_valid = density[valid]
        
        # 创建密度表（直接存密度值，-1 表示空）
        table = torch.full((resolution ** 3,), -1.0, device=self.device)
        table[keys] = density_valid
        
        return table
    
    def query_density_at_level(
        self,
        level_idx: int,
        world_coords: torch.Tensor,  # (..., 3)
    ) -> torch.Tensor:
        """在指定层查询密度"""
        level = self.levels[level_idx]
        shape = world_coords.shape[:-1]
        world_coords = world_coords.reshape(-1, 3)
        
        # 世界坐标 -> 体素坐标
        voxel_size = (self.aabb_max - self.aabb_min) / level.resolution
        voxel_coords = ((world_coords - self.aabb_min) / voxel_size).long()
        
        # 边界检查
        valid = ((voxel_coords >= 0) & (voxel_coords < level.resolution)).all(dim=-1)
        
        # 查询
        res = level.resolution
        keys = (voxel_coords[:, 0] * res * res +
                voxel_coords[:, 1] * res +
                voxel_coords[:, 2]).long()
        keys = keys.clamp(0, res ** 3 - 1)
        
        density = level.hash_table[keys]
        density = torch.where(valid & (density >= 0), density, torch.zeros_like(density))
        
        return density.reshape(shape)
    
    def render_hierarchical(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        H: int = 256,
        W: int = 256,
        coarse_samples: int = 32,
        fine_samples: int = 32,
        near: float = 0.5,
        far: float = 2.5,
    ) -> Dict[str, torch.Tensor]:
        """
        层次化 Ray Marching。
        
        策略:
        1. 在最粗层（subs[0]）采样，找到表面大致位置
        2. 在表面附近，用最细层（subs[-1]）精细采样
        """
        device = self.device
        
        # =====================================================================
        # Step 1: 生成射线
        # =====================================================================
        rays_o, rays_d = self._generate_rays(extrinsics, intrinsics, H, W)
        
        # =====================================================================
        # Step 2: 粗采样（使用最粗层）
        # =====================================================================
        t_vals_coarse = torch.linspace(near, far, coarse_samples, device=device)
        pts_coarse = rays_o.unsqueeze(2) + t_vals_coarse.reshape(1, 1, -1, 1) * rays_d.unsqueeze(2)
        
        # 在粗层查询
        coarse_level = 0  # 使用第 0 层（最粗）
        sigma_coarse = self.query_density_at_level(coarse_level, pts_coarse)  # (H, W, S_coarse)
        
        # 计算粗采样的权重
        delta_coarse = (far - near) / coarse_samples
        alpha_coarse = 1 - torch.exp(-F.relu(sigma_coarse) * delta_coarse * 10)
        T_coarse = torch.cumprod(1 - alpha_coarse + 1e-10, dim=-1)
        T_coarse = torch.cat([torch.ones_like(T_coarse[..., :1]), T_coarse[..., :-1]], dim=-1)
        weights_coarse = T_coarse * alpha_coarse  # (H, W, S_coarse)
        
        # =====================================================================
        # Step 3: 精细采样（在粗权重指导下，使用最细层）
        # =====================================================================
        # 根据粗权重分布采样
        t_vals_fine = self._sample_pdf(
            t_vals_coarse, weights_coarse, fine_samples
        )  # (H, W, S_fine)
        
        # 合并粗细采样点
        t_vals_all, _ = torch.sort(
            torch.cat([t_vals_coarse.unsqueeze(0).unsqueeze(0).expand(H, W, -1), 
                       t_vals_fine], dim=-1),
            dim=-1
        )  # (H, W, S_coarse + S_fine)
        
        pts_all = rays_o.unsqueeze(2) + t_vals_all.unsqueeze(-1) * rays_d.unsqueeze(2)
        
        # 在最细层查询
        fine_level = len(self.levels) - 1
        sigma_fine = self.query_density_at_level(fine_level, pts_all)
        
        # =====================================================================
        # Step 4: 最终积分
        # =====================================================================
        delta_fine = torch.cat([
            t_vals_all[..., 1:] - t_vals_all[..., :-1],
            torch.full_like(t_vals_all[..., :1], 1e-3)
        ], dim=-1)
        
        alpha_fine = 1 - torch.exp(-F.relu(sigma_fine) * delta_fine * 10)
        T_fine = torch.cumprod(1 - alpha_fine + 1e-10, dim=-1)
        T_fine = torch.cat([torch.ones_like(T_fine[..., :1]), T_fine[..., :-1]], dim=-1)
        weights_fine = T_fine * alpha_fine
        
        depth = (weights_fine * t_vals_all).sum(dim=-1)
        alpha = weights_fine.sum(dim=-1)
        
        return {
            'depth': depth,
            'alpha': alpha,
            'weights_coarse': weights_coarse,
            'weights_fine': weights_fine,
        }
    
    def render_multiscale_loss(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        target_mask: torch.Tensor,  # (H, W)
        H: int = 256,
        W: int = 256,
        num_samples: int = 32,
        near: float = 0.5,
        far: float = 2.5,
    ) -> Dict[str, torch.Tensor]:
        """
        多尺度监督：每层独立渲染，计算 loss。
        
        每层都与下采样的 target 对比，让每层 subs 都参与梯度。
        """
        losses = {}
        alphas = {}
        
        rays_o, rays_d = self._generate_rays(extrinsics, intrinsics, H, W)
        t_vals = torch.linspace(near, far, num_samples, device=self.device)
        pts = rays_o.unsqueeze(2) + t_vals.reshape(1, 1, -1, 1) * rays_d.unsqueeze(2)
        
        for level_idx, level in enumerate(self.levels):
            # 该层的渲染分辨率（与体积分辨率成比例）
            scale = level.resolution / self.levels[-1].resolution
            level_H = max(32, int(H * scale))
            level_W = max(32, int(W * scale))
            
            # 下采样射线
            if level_H != H:
                rays_o_level, rays_d_level = self._generate_rays(
                    extrinsics, intrinsics, level_H, level_W
                )
                pts_level = rays_o_level.unsqueeze(2) + \
                            t_vals.reshape(1, 1, -1, 1) * rays_d_level.unsqueeze(2)
            else:
                pts_level = pts
            
            # 查询该层密度
            sigma = self.query_density_at_level(level_idx, pts_level)
            
            # 体渲染
            delta = (far - near) / num_samples
            alpha = 1 - torch.exp(-F.relu(sigma) * delta * 10)
            T = torch.cumprod(1 - alpha + 1e-10, dim=-1)
            T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
            weights = T * alpha
            
            alpha_map = weights.sum(dim=-1)  # (level_H, level_W)
            
            # 下采样 target
            target_level = F.interpolate(
                target_mask.unsqueeze(0).unsqueeze(0).float(),
                size=(level_H, level_W),
                mode='bilinear', align_corners=False
            ).squeeze()
            
            # 计算 loss
            loss = F.mse_loss(alpha_map.clamp(0, 1), target_level)
            
            losses[f'level_{level_idx}'] = loss
            alphas[f'level_{level_idx}'] = alpha_map
        
        # 总 loss（可以加权）
        total_loss = sum(losses.values())
        
        return {
            'total_loss': total_loss,
            'level_losses': losses,
            'level_alphas': alphas,
        }
    
    def _generate_rays(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成射线"""
        device = self.device
        
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        x_cam = (u / W - cx) / fx
        y_cam = (v / H - cy) / fy
        z_cam = torch.ones_like(x_cam)
        
        rays_d_cam = F.normalize(torch.stack([x_cam, y_cam, z_cam], dim=-1), dim=-1)
        
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3]
        rays_o = (-R.T @ t).reshape(1, 1, 3).expand(H, W, 3)
        rays_d = torch.einsum('ij,hwj->hwi', R.T, rays_d_cam)
        
        return rays_o, rays_d
    
    def _sample_pdf(
        self,
        bins: torch.Tensor,      # (S,)
        weights: torch.Tensor,   # (H, W, S)
        num_samples: int,
    ) -> torch.Tensor:
        """根据权重分布采样（Inverse CDF）"""
        H, W, S = weights.shape
        device = weights.device
        
        # 归一化权重
        weights = weights + 1e-5
        pdf = weights / weights.sum(dim=-1, keepdim=True)  # (H, W, S)
        cdf = torch.cumsum(pdf, dim=-1)  # (H, W, S)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (H, W, S+1)
        
        # 均匀采样 [0, 1]
        u = torch.linspace(0, 1, num_samples, device=device)
        u = u.reshape(1, 1, -1).expand(H, W, -1)  # (H, W, num_samples)
        
        # 找到对应的 bin
        indices = torch.searchsorted(cdf, u, right=True)  # (H, W, num_samples)
        indices = indices.clamp(1, S)
        
        # 线性插值
        below = (indices - 1).clamp(0, S - 1)
        above = indices.clamp(0, S - 1)
        
        cdf_below = torch.gather(cdf, -1, below)
        cdf_above = torch.gather(cdf, -1, above)
        bins_below = bins[below.clamp(0, S - 1)]
        bins_above = bins[above.clamp(0, S - 1)]
        
        t = (u - cdf_below) / (cdf_above - cdf_below + 1e-5)
        samples = bins_below + t * (bins_above - bins_below)
        
        return samples  # (H, W, num_samples)
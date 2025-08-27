from typing import *
import torch
import torch.nn as nn
from . import SparseTensor

__all__ = [
    'SparseDownsample',
    'SparseUpsample',
    'SparseSubdivide'
]


class SparseDownsample(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as average pooling.
    """
    def __init__(self, factor: Union[int, Tuple[int, ...], List[int]]):
        super(SparseDownsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1  # 维度数（3D 则为 3）
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the downsample factor.'

        coord = list(input.coords.unbind(dim=-1))  # [b, x, y, z]
        for i, f in enumerate(factor):
            coord[i+1] = coord[i+1] // f

        MAX = [coord[i+1].max().item() + 1 for i in range(DIM)]  # 每维最大坐标+1
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]  # 基数展开偏移
        code = sum([c * o for c, o in zip(coord, OFFSET)])  # 展平编码 (N,)
        code, idx = code.unique(return_inverse=True)  # 去重后的 (N',), 以及映射 idx (N,)

        new_feats = torch.scatter_reduce(
            torch.zeros(code.shape[0], input.feats.shape[1], device=input.feats.device, dtype=input.feats.dtype),  # (N', C)
            dim=0,
            index=idx.unsqueeze(1).expand(-1, input.feats.shape[1]),
            src=input.feats,
            reduce='mean'
        )
        new_coords = torch.stack(
            [code // OFFSET[0]] +
            [(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)],
            dim=-1
        )  # 形状 (N', 1+DIM)
        out = SparseTensor(new_feats, new_coords, input.shape,)  # feats (N', C)
        out._scale = tuple([s // f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.register_spatial_cache(f'upsample_{factor}_coords', input.coords)
        out.register_spatial_cache(f'upsample_{factor}_layout', input.layout)
        out.register_spatial_cache(f'upsample_{factor}_idx', idx)

        return out


class SparseUpsample(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self, factor: Union[int, Tuple[int, int, int], List[int]]):
        super(SparseUpsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the upsample factor.'

        new_coords = input.get_spatial_cache(f'upsample_{factor}_coords')  # (N_up, 1+DIM)
        new_layout = input.get_spatial_cache(f'upsample_{factor}_layout')  # List[slice]
        idx = input.get_spatial_cache(f'upsample_{factor}_idx')  # (N_up,)
        if any([x is None for x in [new_coords, new_layout, idx]]):
            raise ValueError('Upsample cache not found. SparseUpsample must be paired with SparseDownsample.')
        new_feats = input.feats[idx]  # (N_up, C)
        out = SparseTensor(new_feats, new_coords, input.shape, new_layout)  # batched SparseTensor
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        return out
    
class SparseSubdivide(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self):
        super(SparseSubdivide, self).__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        # upsample scale=2^DIM
        n_cube = torch.ones([2] * DIM, device=input.device, dtype=torch.int)  # 2^DIM 个偏移
        n_coords = torch.nonzero(n_cube)
        n_coords = torch.cat([torch.zeros_like(n_coords[:, :1]), n_coords], dim=-1)  # (2^DIM, 1+DIM)
        factor = n_coords.shape[0]
        assert factor == 2 ** DIM
        # print(n_coords.shape)
        new_coords = input.coords.clone()  # (N, 1+DIM)
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(new_coords.dtype)  # (N, 2^DIM, 1+DIM)
        
        new_feats = input.feats.unsqueeze(1).expand(input.feats.shape[0], factor, *input.feats.shape[1:])  # (N, 2^DIM, C)
        out = SparseTensor(new_feats.flatten(0, 1), new_coords.flatten(0, 1), input.shape)  # (N*2^DIM, C), (N*2^DIM, 1+DIM)
        out._scale = input._scale * 2
        out._spatial_cache = input._spatial_cache
        return out


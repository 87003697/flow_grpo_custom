import torch
import torch.nn as nn
from .. import SparseTensor
from .. import DEBUG
from . import SPCONV_ALGO
import spconv.pytorch as spconv

class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        algo = None
        if SPCONV_ALGO == 'native':
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == 'implicit_gemm':
            algo = spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1 and (padding is None):
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
        else:
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.padding = padding

    def _forward_subm_per_sample(self, data):
        # data: SparseConvTensor，features 形状 (N, Cin)
        inds = data.indices  # 形状 (N, 4)
        feats = data.features  # 形状 (N, Cin)
        batch_ids = inds[:, 0]  # 形状 (N,)
        unique_b = torch.unique(batch_ids)  # 形状 (B,)
        out_feats = feats.new_zeros(feats.shape[0], self.conv.out_channels)  # 形状 (N, Cout)

        for b in unique_b:
            mask = (batch_ids == b)  # 形状 (N,)
            idxs = torch.nonzero(mask, as_tuple=False).view(-1)  # 形状 (N_b,)
            sub_feats = feats.index_select(0, idxs)  # 形状 (N_b, Cin)
            sub_inds = inds.index_select(0, idxs)  # 形状 (N_b, 4)
            sub_inds_zero = sub_inds.clone()  # 形状 (N_b, 4)
            sub_inds_zero[:, 0] = 0  # 形状 (N_b, 4)
            sub_data = spconv.SparseConvTensor(sub_feats, sub_inds_zero, data.spatial_shape, 1)  # features 形状 (N_b, Cin)
            new_sub = self.conv(sub_data)  # SparseConvTensor，features 形状 (N_b, Cout)
            out_feats.index_copy_(0, idxs, new_sub.features)  # 形状 (N, Cout)

        return spconv.SparseConvTensor(out_feats, inds, data.spatial_shape, data.batch_size)  # SparseConvTensor，features 形状 (N, Cout)

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)  # bool
        if not spatial_changed:
            new_data = self._forward_subm_per_sample(x.data)  # SparseConvTensor，features 形状 (N, Cout)
        else:
            new_data = self.conv(x.data)  # SparseConvTensor，features 形状 (N', Cout)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
            out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)
 
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride)
        if spatial_changed:
            # recover the original spconv order
            data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
            bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
            data = data.replace_feature(x.feats[bwd])
            if DEBUG:
                assert torch.equal(data.indices, x.coords[bwd]), 'Recover the original order failed'
        else:
            data = x.data

        new_data = self.conv(data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout
        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out

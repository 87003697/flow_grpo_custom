import torch
import os
import timm
import numpy as np
from torch import nn
from . import losses
from pathlib import Path

from .point_encoder import PointcloudEncoder

class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        xyz = pc[:,:,:3].contiguous()
        color = pc[:,:,3:].contiguous()
        pc_feat = self.point_encoder(xyz, color)
        return pc_feat

    def forward(self, pc, text, image):
        text_embed_all = text
        image_embed = image   
        pc_embed = self.encode_pc(pc)
        return {'text_embed': text_embed_all,
                'pc_embed': pc_embed,
                'image_embed': image_embed,
                'logit_scale': self.logit_scale.exp()}

def get_filter_loss(args):
    return losses.Uni3d_Text_Image_Loss()

def get_metric_names(model):
    return ['loss', 'uni3d_loss', 'pc_image_acc', 'pc_text_acc']

def create_uni3d(args):  
    # create transformer blocks for point cloud via timm
    if args.pretrained_pc and Path(args.pretrained_pc).exists():
        if os.environ.get("TRELLIS_VERBOSE", "0") == "1":
            print(f"📁 从本地加载EVA Giant权重: {args.pretrained_pc}")
        # 🔧 使用官方方式：直接通过checkpoint_path参数加载
        try:
            point_transformer = timm.create_model(args.pc_model, 
                                                 checkpoint_path=args.pretrained_pc, 
                                                 drop_path_rate=args.drop_path_rate)
            if os.environ.get("TRELLIS_VERBOSE", "0") == "1":
                print("✅ EVA Giant权重通过checkpoint_path加载成功")
        except Exception as e:
            if os.environ.get("TRELLIS_VERBOSE", "0") == "1":
                print(f"⚠️ checkpoint_path方式失败: {e}")
                print("🔄 回退到手动加载模式...")
            # 回退方案：手动加载
            point_transformer = timm.create_model(args.pc_model, pretrained=False, drop_path_rate=args.drop_path_rate)
            state_dict = torch.load(args.pretrained_pc, map_location='cpu')
            point_transformer.load_state_dict(state_dict, strict=False)
            if os.environ.get("TRELLIS_VERBOSE", "0") == "1":
                print("✅ EVA Giant权重手动加载成功")
    else:
        if os.environ.get("TRELLIS_VERBOSE", "0") == "1":
            print(f"⚠️ EVA Giant权重不存在或未指定本地路径，使用在线下载")
            if args.pretrained_pc:
                print(f"   尝试路径: {args.pretrained_pc}")
            print("💡 运行 python scripts/download_eva_weights.py 来下载权重到本地")
        # 使用在线权重
        point_transformer = timm.create_model(args.pc_model, pretrained=True, drop_path_rate=args.drop_path_rate)

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, args)

    # uni3d model
    model = Uni3D(point_encoder=point_encoder,)
    return model



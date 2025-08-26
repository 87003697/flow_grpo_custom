from typing import Any
import os
import sys
import torch
from PIL import Image
import torchvision.transforms as T
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)
from _reference_codes.VGGTObj.vggt_camera_search.normal_predictor import create_normal_predictor


def prepare_query_tensor(cfg: Any, device: torch.device, image_path: str) -> torch.Tensor:
    """根据 cfg.query_input 构造 VGGT 的 query 图像张量。

    功能:
        - 支持三种输入模式: rgb / normal_pred / normal_image。
        - 在 normal_pred 模式下使用 `_reference_codes/VGGTObj/vggt_camera_search/normal_predictor.py` 创建的本地预测器进行 RGB→Normal 转换。

    输入:
        cfg: 含 `query_input`, `img_size`, `normal_weights_dir`, `normal_version` 等字段的配置对象。
        device: 目标设备。
        image_path: RGB 或已有法线图像路径。
    输出:
        imgs_query: 张量 (1,3,H,W)，H=W=cfg.img_size。

    参考:
        - Normal Predictor: `_reference_codes/VGGTObj/vggt_camera_search/normal_predictor.py` L17-L35, L56-L85
    """
    if cfg.query_input not in {"rgb", "normal_pred", "normal_image"}:
        raise ValueError("query_input 必须为 {'rgb','normal_pred','normal_image'}")

    img_size = int(cfg.img_size)  # 形状: 标量
    if cfg.query_input == "rgb":
        img = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        query_tensor = transform(img).to(device)  # 形状: (3,H,W)
    else:
        if cfg.query_input == "normal_pred":
            predictor = create_normal_predictor(
                weights_dir=cfg.normal_weights_dir,
                yoso_version=cfg.normal_version,
                device=str(device),
            )
            rgb_img = Image.open(image_path).convert("RGB")
            normal_img = predictor.predict(
                rgb_img,
                resolution=img_size,
                match_input_resolution=True,
                data_type="object",
            )
            img_for_tensor = normal_img
        else:  # normal_image
            img_for_tensor = Image.open(image_path).convert("RGB")

        transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        query_tensor = transform(img_for_tensor).to(device)  # 形状: (3,H,W)

    imgs_query = query_tensor.unsqueeze(0)  # 形状: (1,3,H,W)
    return imgs_query



import os
import sys
import torch

# 确保参考代码路径提前注入，再进行下面的顶层导入
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_vggt_root = os.path.join(_proj_root, "_reference_codes", "VGGTObj")
if _vggt_root not in sys.path:
    sys.path.insert(0, _vggt_root)

from safetensors.torch import load_file as load_safetensors
from _reference_codes.VGGTObj.training.models import model_factory as mf
from _reference_codes.VGGTObj.training.models.model_factory import create_model


class PeftConfig:
    def __init__(self) -> None:
        self.method = "lora"
        self.rank = 128
        self.alpha = 128
        self.dropout = 0.1
        self.target_modules = ["qkv", "proj", "fc1", "fc2", "1", "embed_pose"]
        self.bias = "none"


class ModelConfig:
    def __init__(self, img_size: int, embed_dim: int, camera_param_dim: int, use_peft: bool) -> None:
        self.name = "camera_search_vggt"
        self.img_size = int(img_size)
        self.patch_size = 14
        self.embed_dim = int(embed_dim)
        self.enable_camera = True
        self.enable_point = False
        self.enable_depth = False
        self.camera_param_dim = int(camera_param_dim)
        self.use_peft = bool(use_peft)
        self.peft = PeftConfig()


class _DummyPretrained:
    def state_dict(self):
        return {}


def create_vggt_camera_search_model(
    device: torch.device,
    camera_param_dim: int = 9,
    img_size: int = 518,
    ckpt: str | None = None,
    embed_dim: int = 1024,
) -> torch.nn.Module:
    """创建并返回 VGGT Camera-Search 模型（仅启用 camera head）。

    输入:
        device: 目标设备。
        camera_param_dim: 姿态编码维度（默认 9）。
        img_size: 模型期望输入尺寸（默认 518）。
        ckpt: checkpoint 路径或目录（必须提供）。
        embed_dim: 基础 ViT embed 维度。
    输出:
        已加载权重、设置为 eval() 的 torch.nn.Module。
    """
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # 形状: 路径
    vggt_root = os.path.join(proj_root, "_reference_codes", "VGGTObj")  # 形状: 路径
    if vggt_root not in sys.path:
        sys.path.insert(0, vggt_root)

    if ckpt is None or len(str(ckpt)) == 0:
        raise ValueError("必须提供 camera_ckpt（目录或 .safetensors 文件），以对齐 vggt_camera_search 的导入方式")

    ckpt_path = str(ckpt)
    ckpt_file = ckpt_path if ckpt_path.endswith(".safetensors") else os.path.join(ckpt_path, "model.safetensors")
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f"未找到权重文件: {ckpt_file}")

    state = load_safetensors(ckpt_file)
    has_lora_keys = any((".lora_A." in k) or (".lora_B." in k) for k in state.keys())

    if hasattr(mf, "_load_pretrained_vggt"):
        mf._load_pretrained_vggt = lambda: _DummyPretrained()

    cfg = ModelConfig(
        img_size=int(img_size),
        embed_dim=int(embed_dim),
        camera_param_dim=int(camera_param_dim),
        use_peft=bool(has_lora_keys),
    )
    model = create_model(cfg).to(device).eval()

    full_state = model.state_dict()
    has_model_prefix_in_model = any(k.startswith('model.') for k in full_state.keys())
    has_model_prefix_in_ckpt = any(k.startswith('model.') for k in state.keys())
    if (not has_model_prefix_in_ckpt) and has_model_prefix_in_model:
        remapped = {f'model.{k}': v for k, v in state.items()}
    elif has_model_prefix_in_ckpt and (not has_model_prefix_in_model):
        remapped = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in state.items()}
    else:
        remapped = state

    model.load_state_dict(remapped, strict=False)
    return model



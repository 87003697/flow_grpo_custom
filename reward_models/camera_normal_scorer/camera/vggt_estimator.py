from typing import Tuple
import os
import sys
import torch

from reward_models.camera_normal_scorer.camera_estimation import estimate_camera


class VGGTSearchEstimator:
    def __init__(self, device: torch.device, camera_param_dim: int = 9, img_size: int = 518, ckpt: str | None = None, embed_dim: int = 1024) -> None:
        from safetensors.torch import load_file as load_safetensors
        from _reference_codes.VGGTObj.training.models import model_factory as mf
        from _reference_codes.VGGTObj.training.models.model_factory import create_model

        # 确保 vggt 模块可导入
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

        class ModelConfig:
            def __init__(self):
                self.name = "camera_search_vggt"
                self.img_size = int(img_size)
                self.patch_size = 14
                self.embed_dim = int(embed_dim)
                self.enable_camera = True
                # 仅保留 camera_head，关闭 point/depth 两个分支以降显存
                self.enable_point = False
                # 删除 depth head：不创建 depth 分支
                self.enable_depth = False
                self.camera_param_dim = int(camera_param_dim)
                # 删除所有与 depth head 相关的配置项
                self.use_peft = bool(has_lora_keys)
                class Peft:
                    def __init__(self):
                        self.method = "lora"
                        self.rank = 128
                        self.alpha = 128
                        self.dropout = 0.1
                        self.target_modules = ["qkv", "proj", "fc1", "fc2", "1", "embed_pose"]
                        self.bias = "none"
                self.peft = Peft()

        # 离线环境：屏蔽 _load_pretrained_vggt 权重下载与迁移
        class _DummyPretrained:
            def state_dict(self):
                return {}
        if hasattr(mf, "_load_pretrained_vggt"):
            mf._load_pretrained_vggt = lambda: _DummyPretrained()

        cfg = ModelConfig()
        model = create_model(cfg).to(device).eval()

        def _merge_state(model_state: dict, src_state: dict) -> tuple[int, dict]:
            merged = dict(model_state)
            cnt = 0
            for k, v in src_state.items():
                if (k in merged) and (merged[k].shape == v.shape):
                    merged[k] = v
                    cnt += 1
            return cnt, merged

        full_state = model.state_dict()
        has_model_prefix_in_model = any(k.startswith('model.') for k in full_state.keys())
        has_model_prefix_in_ckpt = any(k.startswith('model.') for k in state.keys())
        if (not has_model_prefix_in_ckpt) and has_model_prefix_in_model:
            remapped = {f'model.{k}': v for k, v in state.items()}
        elif has_model_prefix_in_ckpt and (not has_model_prefix_in_model):
            remapped = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in state.items()}
        else:
            remapped = state
        # 宽松加载，避免与 depth 分支或LoRA键不匹配导致中止
        model.load_state_dict(remapped, strict=False)

        self.model = model
        self.device = device

    @torch.no_grad()
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        from vggt.utils.load_fn import load_and_preprocess_images  # 形状: 可调用
        return load_and_preprocess_images([image_path], mode="crop").to(self.device)  # 形状: (1,3,H,W)

    @torch.no_grad()
    def estimate(self, images_batched: torch.Tensor, support: torch.Tensor, image_hw: Tuple[int, int]):
        extri_4x4, intr_3x3 = estimate_camera(images_batched, support, self.model, image_hw)  # 形状: (B,4,4),(B,3,3)
        return extri_4x4, intr_3x3  # 形状: (B,4,4),(B,3,3)



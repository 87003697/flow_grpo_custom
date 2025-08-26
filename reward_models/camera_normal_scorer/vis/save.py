import os
import torch
from PIL import Image


def save_similarity_inputs(n_img: torch.Tensor, n_mesh: torch.Tensor, vis_dir: str, tag: str) -> None:
    os.makedirs(vis_dir, exist_ok=True)
    Image.fromarray(((n_img.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()).save(
        os.path.join(vis_dir, f"pred_normal_{tag}.png")
    )
    Image.fromarray(((n_mesh.clamp(-1, 1) + 1.0) * 0.5 * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()).save(
        os.path.join(vis_dir, f"render_normal_{tag}.png")
    )



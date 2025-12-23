import math
import torch


def _normalize_fovy_to_batch(
    fovy_deg: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # 将 fovy 统一为 [B]
    if not torch.is_tensor(fovy_deg):
        fovy_t = torch.full((batch_size,), float(fovy_deg), dtype=dtype, device=device)  # [B]
    else:
        if fovy_deg.dim() == 0:
            fovy_t = torch.full((batch_size,), float(fovy_deg.item()), dtype=dtype, device=device)  # [B]
        else:
            fovy_t = fovy_deg.to(device=device, dtype=dtype).reshape(-1)  # [B]
    return fovy_t  # [B]


def build_perspective_matrix(
    fovy_deg: torch.Tensor,
    aspect: float,
    znear: float,
    zfar: float,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """构造批量透视投影矩阵。

    参数:
    - fovy_deg: [] | [B]
    - aspect: 标量宽高比
    - znear, zfar: 标量近远裁剪面
    - dtype, device: 张量类型与设备
    - batch_size: 期望批量大小（用于将标量 fovy 扩展为 [B]）

    返回:
    - proj: [B,4,4]
    """
    fovy_t = _normalize_fovy_to_batch(fovy_deg, batch_size, dtype, device)  # [B]

    f = 1.0 / torch.tan((fovy_t * math.pi / 180.0) * 0.5)  # [B]
    A = torch.tensor(aspect, dtype=dtype, device=device)  # []
    zn = torch.tensor(znear, dtype=dtype, device=device)  # []
    zf = torch.tensor(zfar, dtype=dtype, device=device)  # []

    proj = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)  # [B,4,4]
    proj[:, 0, 0] = f / A  # [B]
    # 与 nvdiffrast 约定对齐：Y 轴取负，避免渲染结果上下颠倒
    proj[:, 1, 1] = -f  # [B]
    proj[:, 2, 2] = (zf + zn) / (zn - zf)  # [B]
    proj[:, 2, 3] = (2.0 * zf * zn) / (zn - zf)  # [B]
    proj[:, 3, 2] = -1.0  # [B]
    return proj  # [B,4,4]


def build_mvp_from_w2c(
    w2c: torch.Tensor,
    proj: torch.Tensor,
) -> torch.Tensor:
    """根据 w2c 与投影矩阵构造 MVP。

    支持两种 w2c 形状：
    - [B,4,4]
    - [B,V,4,4]

    要求 proj 形状为 [B,4,4]。
    返回与 w2c 同维度的 mvp。
    """
    if w2c.dim() == 3:
        mvp = torch.matmul(proj, w2c)  # [B,4,4]
        return mvp  # [B,4,4]
    if w2c.dim() == 4:
        mvp = torch.matmul(proj[:, None, ...], w2c)  # [B,V,4,4]
        return mvp  # [B,V,4,4]
    raise ValueError("w2c must be [B,4,4] or [B,V,4,4]")


def build_w2c_from_c2w(c2w: torch.Tensor) -> torch.Tensor:
    """从 c2w 计算 w2c，支持 [B,4,4] 或 [B,V,4,4]。

    注意：不假设严格正交，直接使用矩阵逆。
    """
    if c2w.dim() == 3:
        w2c = torch.inverse(c2w)  # [B,4,4]
        return w2c  # [B,4,4]
    if c2w.dim() == 4:
        B, V = c2w.shape[:2]  # [], []
        c2w_flat = c2w.reshape(B * V, 4, 4)  # [B*V,4,4]
        w2c_flat = torch.inverse(c2w_flat)  # [B*V,4,4]
        w2c = w2c_flat.reshape(B, V, 4, 4)  # [B,V,4,4]
        return w2c  # [B,V,4,4]
    raise ValueError("c2w must be [B,4,4] or [B,V,4,4]")


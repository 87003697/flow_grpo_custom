#!/usr/bin/env python3
"""
准备 GSO (Google Scanned Objects) 测试数据集。

数据来源：
    Roldbach/google_scanned_objects (HuggingFace)
    - 1030 个物体，每个物体 25 个视角（Blender 渲染，RGBA 白底，512×512）
    - 相机位姿为随机 MVS 布局（非固定方位角），world-to-camera 矩阵存 .npy

视角选择策略（对齐 Zero123 评测协议）：
    Roldbach 每个物体的 25 个视角是随机分布的，因此对每个目标方位角
    选取得分最低的视角：score = |elev - target_elev| + 0.5 * |Δazim|（单位：度）。
    注：Blender 默认"正面视图"（Numpad 1）相机在 -Y 轴，atan2(y,x)=-90°。
    主参考图目标：elev=30°, azim=-90°（正面）。

输出文件命名：
    gso_{i:03d}.png              ← 主参考图（azim ≈ 90°）
    gso_{i:03d}_az{target}.png   ← 额外对比视角（--extra_azims 指定，默认 45 135）
    例：gso_000.png, gso_000_az045.png, gso_000_az135.png

功能：
1. 从 HuggingFace 下载 Roldbach/google_scanned_objects（单个 zip，~4.7 GB）
2. 解压到 /data/zhiyuan_ma/data/gso_extracted/
3. 打印 zip 内部结构（用于 --inspect 模式）
4. 采样 100 个物体，每个选最接近各目标方位角的视角（RGBA → 白底 RGBA）
5. 输出到 /data/zhiyuan_ma/data/gso_test/
6. 在代码目录创建 dataset/gso_test 软链接

用法：
    # 1. 先检查内部结构（不提取图片）
    python scripts/download/prepare_gso_dataset.py --inspect

    # 2. 完整运行（下载 + 解压 + 提取，含默认对比视角）
    python scripts/download/prepare_gso_dataset.py

    # 3. 已有 zip 时跳过下载
    python scripts/download/prepare_gso_dataset.py --local_zip /data/zhiyuan_ma/data/gso/google_scanned_objects.zip

    # 4. 自定义额外对比方位角（0=侧面，45，135，180=背面）
    python scripts/download/prepare_gso_dataset.py --extra_azims 0 45 135 180
"""

import argparse
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# =====================================================================
# 配置
# =====================================================================

HF_REPO_ID = "Roldbach/google_scanned_objects"
HF_FILENAME = "google_scanned_objects.zip"

GSO_DOWNLOAD_DIR = Path("/data/zhiyuan_ma/data/gso")
GSO_EXTRACT_DIR = Path("/data/zhiyuan_ma/data/gso_extracted")
GSO_TEST_DIR = Path("/data/zhiyuan_ma/data/gso_test")

N_OBJECTS = 100
RANDOM_SEED = 42

# 主参考图目标视角
# Blender 默认"正面视图"（Numpad 1）相机在 -Y 轴，atan2(y,x)=-90°
TARGET_ELEV = 30.0
TARGET_AZIM = -90.0

# 默认额外对比方位角（±45° 偏转，绝对值）
DEFAULT_EXTRA_AZIMS: list[int] = [45, 135]


# =====================================================================
# 工具函数
# =====================================================================

def rgba_to_white_rgb(im: Image.Image) -> Image.Image:
    """将 RGBA 图合成白色背景，返回 RGB。"""
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGB"), mask=im.split()[3])
        return bg
    return im.convert("RGB")


def find_object_dirs(extract_root: Path) -> list[Path]:
    """在解压目录中寻找物体子目录。

    已知 Roldbach/google_scanned_objects zip 的深层结构：
        <extract_root>/.../google_scanned_blender_25_w2c/<object>/render_mvs_25/model/000.png

    策略：
    1. 先找包含 render_mvs_25 子目录的目录层（物体根目录）
    2. 若未找到，退回通用逻辑（逐层找包含 PNG 的最浅层）
    """
    # --- 方法 1：找 render_mvs_25 模式（GSO 特定结构）---
    # 查找所有 render_mvs_25 目录，其父目录即为物体目录
    obj_dirs = set()
    for mvs_dir in extract_root.rglob("render_mvs_25"):
        if mvs_dir.is_dir():
            obj_dirs.add(mvs_dir.parent)
    if obj_dirs:
        return sorted(obj_dirs)

    # --- 方法 2：通用回退 —— 找包含 PNG 的最浅子目录 ---
    top_dirs = sorted([d for d in extract_root.iterdir() if d.is_dir()])
    if not top_dirs:
        return [extract_root]

    # 若根下直接有物体目录（每个含 PNG）
    sample_pngs = list(top_dirs[0].rglob("*.png"))[:1]
    if sample_pngs:
        return top_dirs

    # 再往下一层
    nested = []
    for d in top_dirs:
        nested.extend(sorted([x for x in d.iterdir() if x.is_dir()]))
    return nested if nested else top_dirs


def _parse_w2c(npy_path: Path):
    """从 .npy 加载 world-to-camera (3×4) 矩阵，返回 (elevation_deg, azimuth_deg)。

    计算方式：
        相机在世界坐标系中的位置 = -R^T @ t
        elevation = arcsin(z / r)
        azimuth   = atan2(y, x)   （Blender 坐标系：Z 向上，Y 向前；正面相机在 azim≈90°）
    """
    m = np.load(str(npy_path))          # shape (3, 4)
    R, t = m[:3, :3], m[:3, 3]
    cam_pos = -R.T @ t                  # 世界坐标系中相机位置
    r = float(np.linalg.norm(cam_pos))
    if r < 1e-6:
        return 0.0, 0.0
    elev = float(np.degrees(np.arcsin(np.clip(cam_pos[2] / r, -1, 1))))
    azim = float(np.degrees(np.arctan2(cam_pos[1], cam_pos[0])))
    return elev, azim


def pick_best_for_azim(
    obj_dir: Path,
    target_azim: float,
    target_elev: float = TARGET_ELEV,
) -> tuple[Optional[Path], float]:
    """从物体目录中挑选最接近指定 (elev, azim) 目标的视角图。

    Roldbach GSO 结构：render_mvs_25/model/{000..024}.png + {000..024}.npy（w2c 矩阵）

    评分 = |elev - target_elev| + 0.5 * |Δazim|（单位：度），取最小值对应的视角。
    方位角差值折叠到 [-180°, 180°)。

    若无 .npy 相机矩阵（非 GSO 格式），退回到编号最小的 PNG，score=inf。

    返回 (png_path, best_score)；未找到任何图片时返回 (None, inf)。
    """
    model_dir = obj_dir / "render_mvs_25" / "model"

    if model_dir.is_dir():
        npys = sorted(model_dir.glob("*.npy"))
        if npys:
            best_png, best_score = None, float("inf")
            for npy_path in npys:
                png_path = npy_path.with_suffix(".png")
                if not png_path.exists():
                    continue
                try:
                    elev, azim = _parse_w2c(npy_path)
                except Exception:
                    continue
                azim_diff = abs(((azim - target_azim) + 180) % 360 - 180)
                score = abs(elev - target_elev) + 0.5 * azim_diff
                if score < best_score:
                    best_score, best_png = score, png_path
            if best_png is not None:
                return best_png, best_score

        pngs = sorted(model_dir.glob("*.png"))
        if pngs:
            return pngs[0], float("inf")

    # --- 通用回退（非 GSO 格式）---
    for sub in ["renders", "images", "render", "image"]:
        sub_dir = obj_dir / sub
        if sub_dir.is_dir():
            pngs = sorted(sub_dir.glob("*.png"))
            if pngs:
                return pngs[0], float("inf")

    candidates = sorted(p for p in obj_dir.rglob("*.png"))
    if candidates:
        return candidates[0], float("inf")
    return None, float("inf")


# =====================================================================
# 主逻辑
# =====================================================================

def download_zip(local_zip: Optional[Path]) -> Path:
    """下载或使用本地 zip，返回 zip 路径。"""
    if local_zip is not None:
        local_zip = Path(local_zip)
        if not local_zip.exists():
            print(f"[ERROR] 指定的本地 zip 不存在: {local_zip}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] 使用本地 zip: {local_zip}")
        return local_zip

    GSO_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = GSO_DOWNLOAD_DIR / HF_FILENAME

    if target.exists():
        print(f"[INFO] zip 已存在，跳过下载: {target}")
        return target

    print(f"[INFO] 开始从 HuggingFace 下载 {HF_REPO_ID}/{HF_FILENAME} ...")
    print("[INFO] 大约 4.7 GB，请耐心等待...")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir=str(GSO_DOWNLOAD_DIR),
        )
        print(f"[INFO] 下载完成: {downloaded}")
        return Path(downloaded)
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}", file=sys.stderr)
        print("[HINT] 可以手动下载后用 --local_zip 参数指定路径", file=sys.stderr)
        sys.exit(1)


def inspect_zip(zip_path: Path, n: int = 30):
    """打印 zip 前 n 条内部路径，用于了解目录结构。"""
    print(f"\n[INSPECT] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    print(f"[INSPECT] 共 {len(names)} 个条目，前 {n} 条：")
    for name in names[:n]:
        print(f"  {name}")
    if len(names) > n:
        print(f"  ... (共 {len(names)} 条)")


def extract_zip(zip_path: Path, extract_dir: Path):
    """解压 zip 到 extract_dir（若已存在则跳过）。"""
    marker = extract_dir / ".extracted"
    if marker.exists():
        print(f"[INFO] 已解压，跳过: {extract_dir}")
        return

    print(f"[INFO] 正在解压到 {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))
    marker.touch()
    print("[INFO] 解压完成")


def _save_rgba(img_path: Path, out_path: Path) -> bool:
    """保存 RGBA 图片（保留透明通道）。失败返回 False。"""
    try:
        im = Image.open(img_path).convert("RGBA")
        im.save(str(out_path))
        return True
    except Exception as e:
        print(f"[WARN] 处理失败 {img_path}: {e}")
        return False


def prepare_test_set(
    extract_dir: Path,
    out_dir: Path,
    n: int,
    seed: int,
    extra_azims: list[int] | None = None,
):
    """从解压目录采样 n 个物体，提取参考图及对比视角到 out_dir。

    输出文件：
        gso_{i:03d}.png              ← 主参考图（azim≈TARGET_AZIM=90°）
        gso_{i:03d}_az{target}.png   ← 每个 extra_azims 中的目标方位角各一张
    """
    if extra_azims is None:
        extra_azims = DEFAULT_EXTRA_AZIMS

    out_dir.mkdir(parents=True, exist_ok=True)

    obj_dirs = find_object_dirs(extract_dir)
    print(f"[INFO] 发现 {len(obj_dirs)} 个物体目录")

    if len(obj_dirs) < n:
        print(f"[WARN] 物体数量 ({len(obj_dirs)}) 少于请求数量 ({n})，使用全部物体")
        n = len(obj_dirs)

    random.seed(seed)
    selected = random.sample(obj_dirs, n)
    selected.sort(key=lambda p: p.name)

    # 汇总所有要保存的方位角：主参考图 + 额外对比
    all_azims: list[tuple[float, str]] = [(TARGET_AZIM, "")]  # (目标azim, 文件名后缀)
    for az in extra_azims:
        all_azims.append((float(az), f"_az{az:03d}"))

    success_main, fail = 0, 0
    for i, obj_dir in enumerate(selected):
        obj_ok = False
        for target_azim, suffix in all_azims:
            img_path, score = pick_best_for_azim(obj_dir, target_azim)
            if img_path is None:
                if suffix == "":
                    print(f"[WARN] 未找到图片: {obj_dir.name}")
                continue
            out_path = out_dir / f"gso_{i:03d}{suffix}.png"
            ok = _save_rgba(img_path, out_path)
            if suffix == "":
                if ok:
                    success_main += 1
                    obj_ok = True
                else:
                    fail += 1
        if not obj_ok and not any(
            (out_dir / f"gso_{i:03d}.png").exists() for _ in [1]
        ):
            fail += 1

    total_extras = len(extra_azims)
    print(f"[INFO] 提取完成：主参考图 {success_main} 张，失败 {fail} 张")
    if total_extras:
        print(f"[INFO] 额外对比视角：每物体最多 {total_extras} 个"
              f"（目标方位角 {extra_azims}°）")
    print(f"[INFO] 输出目录：{out_dir}")
    return success_main


def create_symlink(target: Path, link_name: Path):
    """在代码目录创建软链接。若已存在则跳过。"""
    if link_name.exists() or link_name.is_symlink():
        print(f"[INFO] 软链接已存在，跳过: {link_name}")
        return
    link_name.parent.mkdir(parents=True, exist_ok=True)
    link_name.symlink_to(target)
    print(f"[INFO] 创建软链接: {link_name} -> {target}")


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="准备 GSO 测试数据集")
    parser.add_argument(
        "--local_zip", type=str, default=None,
        help="本地 zip 路径（跳过下载）",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="只打印 zip 内部结构，不提取图片",
    )
    parser.add_argument(
        "--n", type=int, default=N_OBJECTS,
        help=f"采样物体数量（默认 {N_OBJECTS}）",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"随机种子（默认 {RANDOM_SEED}）",
    )
    parser.add_argument(
        "--extract_dir", type=str, default=str(GSO_EXTRACT_DIR),
        help=f"解压目录（默认 {GSO_EXTRACT_DIR}）",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(GSO_TEST_DIR),
        help=f"输出测试集目录（默认 {GSO_TEST_DIR}）",
    )
    parser.add_argument(
        "--symlink", type=str,
        default=str(Path(__file__).resolve().parents[2] / "dataset" / "gso_test"),
        help="在代码目录创建的软链接路径",
    )
    parser.add_argument(
        "--extra_azims", type=int, nargs="*", default=DEFAULT_EXTRA_AZIMS,
        metavar="AZ",
        help=(
            f"额外保存的对比方位角列表（绝对值，单位：度，默认 {DEFAULT_EXTRA_AZIMS}）。"
            " 主参考图始终以 azim=-90° 保存为 gso_NNN.png，"
            " 对比图以 gso_NNN_azXXX.png 保存。"
            " 传空列表（--extra_azims）可禁用对比视角。"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    zip_path = download_zip(args.local_zip)

    if args.inspect:
        inspect_zip(zip_path)
        return

    # 先打印结构供参考
    inspect_zip(zip_path, n=20)

    extract_dir = Path(args.extract_dir)
    extract_zip(zip_path, extract_dir)

    extra_azims: list[int] = args.extra_azims if args.extra_azims else []
    out_dir = Path(args.out_dir)
    n_success = prepare_test_set(
        extract_dir, out_dir, args.n, args.seed, extra_azims=extra_azims
    )

    if n_success > 0:
        create_symlink(out_dir, Path(args.symlink))
        print(f"\n[DONE] GSO 测试集已准备好：{out_dir}（主参考图 {n_success} 张）")
        if extra_azims:
            names = ", ".join(f"gso_NNN_az{az:03d}.png" for az in extra_azims)
            print(f"[DONE] 对比视角：{names}")
        print(f"[DONE] 软链接：{args.symlink}")
    else:
        print("[ERROR] 未能提取任何图片，请检查 zip 内容", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

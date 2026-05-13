#!/usr/bin/env python3
"""
准备 Toys4k 单视图测试数据集（对齐 GSO 格式）。

来源：
    ZIP 文件由 https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k
    提供，结构为 toys4k_blend_files/<category>/<instance>/<instance>.blend

输出：
    512×512 RGBA PNG，透明背景，格式为 toys4k_000.png …
    相机：仰角 30°，方位角 0°，半径 2，FOV 40°（与 TRELLIS 训练时一致）

用法：
    # 完整流程（解压 + 渲染）
    bash scripts/download/prepare_toys4k_dataset.sh

    # 只解压，不渲染
    python scripts/download/prepare_toys4k_dataset.py --extract_only

    # 已解压时跳过解压（直接渲染）
    python scripts/download/prepare_toys4k_dataset.py --skip_extract

    # 只渲染前 N 个（调试）
    python scripts/download/prepare_toys4k_dataset.py --max_objects 10

    # 多进程并行（每个 worker 调用一次 Blender）
    python scripts/download/prepare_toys4k_dataset.py --workers 8
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# =====================================================================
# 路径配置
# =====================================================================

BLENDER_PATH = "/data/zhiyuan_ma/tools/blender-4.2.0-linux-x64/blender"
BLENDER_SCRIPT = str(
    Path(__file__).parents[2]
    / "_reference_codes/TRELLIS/dataset_toolkits/blender_script/render.py"
)

ZIP_PATH = Path("/data/zhiyuan_ma/code/flow_grpo_custom/dataset/Toys4k/raw/toys4k_blend_files.zip")
EXTRACT_DIR = Path("/data/zhiyuan_ma/code/flow_grpo_custom/dataset/Toys4k/raw/toys4k_blend_files")
OUTPUT_DIR = Path("/data/zhiyuan_ma/data/toys4k_test")
SYMLINK_DIR = Path("/data/zhiyuan_ma/code/flow_grpo_custom/dataset/toys4k_test")

# 相机参数（对齐 TRELLIS 训练；仰角 30°，方位角 0°）
SINGLE_VIEW = [{"yaw": -math.pi / 2, "pitch": math.radians(30), "radius": 2.0, "fov": math.radians(40)}]

RENDER_RESOLUTION = 512

# =====================================================================
# 工具函数
# =====================================================================

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    print(f"解压 {zip_path} → {extract_to} …")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".blend")]
        for member in tqdm(members, desc="解压 .blend"):
            zf.extract(member, extract_to.parent)
    print(f"解压完成，共 {len(members)} 个 .blend 文件")


def find_blend_files(extract_dir: Path) -> list[Path]:
    """返回排序后的 .blend 文件列表。"""
    files = sorted(extract_dir.rglob("*.blend"))
    return files


def render_one(args_tuple):
    """在子进程中调用 Blender 渲染单张图（供 multiprocessing 使用）。"""
    blend_file, tmp_dir, output_path, blender_path, blender_script = args_tuple

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    views_json = json.dumps(SINGLE_VIEW)

    cmd = [
        blender_path,
        str(blend_file),          # Blender 以 .blend 为项目文件打开
        "-b",
        "-P", blender_script,
        "--",
        "--views", views_json,
        "--object", str(blend_file),   # 告知脚本这是 .blend，跳过 load_object
        "--resolution", str(RENDER_RESOLUTION),
        "--output_folder", str(tmp_dir),
        "--engine", "CYCLES",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=300,
    )

    # 渲染成功时，Blender 输出 000.png
    rendered = tmp_dir / "000.png"
    if rendered.exists():
        shutil.move(str(rendered), str(output_path))
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return str(output_path), None
    else:
        err = result.stderr.decode(errors="replace")[-500:]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, f"{blend_file.name}: {err}"


# =====================================================================
# 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="准备 Toys4k 单视图测试数据集")
    parser.add_argument("--zip_path", type=str, default=str(ZIP_PATH))
    parser.add_argument("--extract_dir", type=str, default=str(EXTRACT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--extract_only", action="store_true", help="只解压，不渲染")
    parser.add_argument("--skip_extract", action="store_true", help="跳过解压（已解压）")
    parser.add_argument("--max_objects", type=int, default=None, help="最多渲染前 N 个物体（调试）")
    parser.add_argument("--workers", type=int, default=4, help="并行 Blender 进程数")
    parser.add_argument("--resume", action="store_true", help="跳过已存在的输出文件")
    opt = parser.parse_args()

    zip_path = Path(opt.zip_path)
    extract_dir = Path(opt.extract_dir)
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 解压
    if not opt.skip_extract:
        if not extract_dir.exists() or not any(extract_dir.rglob("*.blend")):
            extract_zip(zip_path, extract_dir)
        else:
            print(f"已发现解压目录 {extract_dir}，跳过解压")
    else:
        print(f"--skip_extract：跳过解压，使用 {extract_dir}")

    if opt.extract_only:
        print("--extract_only：解压完成，退出")
        return

    # Step 2: 收集 .blend 文件
    blend_files = find_blend_files(extract_dir)
    print(f"找到 {len(blend_files)} 个 .blend 文件")

    if opt.max_objects is not None:
        blend_files = blend_files[: opt.max_objects]
        print(f"限制为前 {len(blend_files)} 个（--max_objects）")

    # Step 3: 构建任务列表
    tasks = []
    for idx, blend_file in enumerate(blend_files):
        out_name = f"toys4k_{idx:04d}.png"
        out_path = output_dir / out_name
        if opt.resume and out_path.exists():
            continue
        tmp_dir = output_dir / f".tmp_{idx:04d}"
        tasks.append((blend_file, tmp_dir, out_path, BLENDER_PATH, BLENDER_SCRIPT))

    if not tasks:
        print("所有文件已渲染，无需重新处理（--resume）")
    else:
        print(f"待渲染：{len(tasks)} 个（workers={opt.workers}）")

    # Step 4: 并行渲染
    success, errors = 0, []
    with ProcessPoolExecutor(max_workers=opt.workers) as executor:
        futures = {executor.submit(render_one, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="渲染") as pbar:
            for future in as_completed(futures):
                out_path, err = future.result()
                if out_path:
                    success += 1
                else:
                    errors.append(err)
                pbar.update(1)
                pbar.set_postfix(ok=success, fail=len(errors))

    print(f"\n渲染完成：成功 {success}，失败 {len(errors)}")
    if errors:
        error_log = output_dir / "render_errors.txt"
        with open(error_log, "w") as f:
            f.write("\n".join(errors))
        print(f"错误详情见 {error_log}")

    # Step 5: 保存 index CSV
    all_pngs = sorted(output_dir.glob("toys4k_*.png"))
    records = []
    for png in all_pngs:
        idx = int(re.search(r"(\d+)", png.stem).group(1))
        blend_file = blend_files[idx] if idx < len(blend_files) else None
        category = blend_file.parts[-3] if blend_file else ""
        instance = blend_file.stem if blend_file else ""
        records.append({"filename": png.name, "category": category, "instance": instance})
    csv_path = output_dir / "index.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"索引保存至 {csv_path}")

    # Step 6: 创建软链接
    symlink = SYMLINK_DIR
    if not symlink.exists():
        symlink.symlink_to(output_dir)
        print(f"软链接：{symlink} → {output_dir}")
    else:
        print(f"软链接已存在：{symlink}")

    print(f"\n完成！共 {len(all_pngs)} 张图像保存在 {output_dir}")


if __name__ == "__main__":
    main()

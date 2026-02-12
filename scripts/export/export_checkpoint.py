#!/usr/bin/env python
"""
将训练 checkpoint 导出为 TRELLIS 推理兼容的权重目录。

导出后可直接用于推理：
    pipe = TrellisImageTo3DPipeline.from_pretrained("exports/my_model")

用法：
    python scripts/export/export_checkpoint.py \
        --checkpoint logs/.../checkpoints/checkpoint_0_100 \
        --output exports/my_finetuned_trellis \
        --pretrained pretrained_weights/TRELLIS-image-large
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export training checkpoint to TRELLIS inference format")
    parser.add_argument("--checkpoint", required=True, help="训练 checkpoint 目录")
    parser.add_argument("--output", required=True, help="导出目标目录")
    parser.add_argument("--pretrained", default="pretrained_weights/TRELLIS-image-large",
                        help="原始预训练权重目录（用于补全未训练的模型）")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    export_dir = Path(args.output)
    pretrained_dir = Path(args.pretrained)

    # ---- 校验 ----
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint 目录不存在: {ckpt_dir}")
    if not (pretrained_dir / "pipeline.json").exists():
        raise FileNotFoundError(f"预训练目录缺少 pipeline.json: {pretrained_dir}")

    # ---- 读取 pipeline.json 获取模型名映射 ----
    with open(pretrained_dir / "pipeline.json") as f:
        pipeline_cfg = json.load(f)

    # 内部名 → TRELLIS ckpt 相对路径
    # e.g. {"slat_flow_model": "ckpts/slat_flow_img_dit_L_64l8p2_fp16", ...}
    model_mapping = pipeline_cfg["args"]["models"]

    # ---- 准备导出目录 ----
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)
    (export_dir / "ckpts").mkdir()

    # 复制 pipeline.json
    shutil.copy2(pretrained_dir / "pipeline.json", export_dir / "pipeline.json")
    logger.info(f"[EXPORT] pipeline.json → {export_dir / 'pipeline.json'}")

    # ---- 逐模型处理 ----
    for internal_name, ckpt_rel_path in model_mapping.items():
        # 复制 config json（模型结构定义）
        src_json = pretrained_dir / f"{ckpt_rel_path}.json"
        dst_json = export_dir / f"{ckpt_rel_path}.json"
        if src_json.exists():
            shutil.copy2(src_json, dst_json)

        # 检查 checkpoint 里是否有此模型的训练权重
        # accelerator.save_state() 保存为 model.safetensors（单模型）
        # 或 model_0.safetensors, model_1.safetensors（多模型，按 prepare 注册顺序）
        trained_weights = ckpt_dir / "model.safetensors"
        src_weights = pretrained_dir / f"{ckpt_rel_path}.safetensors"
        dst_weights = export_dir / f"{ckpt_rel_path}.safetensors"

        if internal_name == "slat_flow_model" and trained_weights.exists():
            # ✅ 用训练后的权重替换
            shutil.copy2(trained_weights, dst_weights)
            logger.info(f"[EXPORT] {internal_name}: ✅ 使用训练权重 ← {trained_weights}")
        elif src_weights.exists():
            # 📋 用原始预训练权重
            shutil.copy2(src_weights, dst_weights)
            logger.info(f"[EXPORT] {internal_name}: 📋 使用原始权重 ← {src_weights}")
        else:
            logger.warning(f"[EXPORT] {internal_name}: ⚠️ 无权重文件！(checkpoint 和 pretrained 都没有)")

    # ---- 复制 meta.json ----
    meta_src = ckpt_dir / "meta.json"
    if meta_src.exists():
        shutil.copy2(meta_src, export_dir / "meta.json")
        with open(meta_src) as f:
            meta = json.load(f)
        logger.info(f"[EXPORT] 来源 checkpoint: epoch={meta['epoch']}, step={meta['global_step']}")

    logger.info(f"[EXPORT] ✅ 导出完成 → {export_dir}")
    logger.info(f"[EXPORT] 使用方式: TrellisImageTo3DPipeline.from_pretrained('{export_dir}')")


if __name__ == "__main__":
    main()

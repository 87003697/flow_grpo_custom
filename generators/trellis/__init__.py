#!/usr/bin/env python3
"""
TRELLIS Stage 2 GRPO 训练生成器模块
"""

from .pipeline import TrellisStage2Pipeline
from .utils import trellis_preprocess_image, convert_trellis_to_trimesh

__all__ = [
    'TrellisStage2Pipeline',
    'trellis_preprocess_image', 
    'convert_trellis_to_trimesh'
] 
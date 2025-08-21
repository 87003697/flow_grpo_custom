#!/usr/bin/env python3
"""
TRELLIS Stage 2 GRPO 训练生成器模块
门面导出：sparse 与 TrellisImageTo3DPipeline
"""
import sys

from .modules import sparse  # 先导出 sparse，供 utils/pipeline 使用，避免循环
# 将本地 sparse 及其子模块映射到参考路径，保证双方共享同一类定义
sys.modules['trellis.modules.sparse'] = sparse
try:
    from .modules.sparse import basic as _s_basic
    from .modules.sparse import norm as _s_norm
    from .modules.sparse import nonlinearity as _s_nlin
    from .modules.sparse import linear as _s_linear
    from .modules.sparse import attention as _s_attn
    from .modules.sparse import conv as _s_conv
    from .modules.sparse import spatial as _s_spatial
    from .modules.sparse import transformer as _s_trans
    sys.modules['trellis.modules.sparse.basic'] = _s_basic
    sys.modules['trellis.modules.sparse.norm'] = _s_norm
    sys.modules['trellis.modules.sparse.nonlinearity'] = _s_nlin
    sys.modules['trellis.modules.sparse.linear'] = _s_linear
    sys.modules['trellis.modules.sparse.attention'] = _s_attn
    sys.modules['trellis.modules.sparse.conv'] = _s_conv
    sys.modules['trellis.modules.sparse.spatial'] = _s_spatial
    sys.modules['trellis.modules.sparse.transformer'] = _s_trans
except Exception:
    pass

from . import models  # 确保 models 可用于 pipelines.base 的相对导入
from .pipelines import TrellisImageTo3DPipeline
from .pipeline import TrellisStage2Pipeline

__all__ = [
    'TrellisStage2Pipeline',
    'sparse',
    'TrellisImageTo3DPipeline'
]
"""
Guidance 后端实现。

目前只支持 LocalGuidance（同进程多 GPU）。
"""

from edit4shape.guidance.backends.local import LocalGuidance

__all__ = ["LocalGuidance"]

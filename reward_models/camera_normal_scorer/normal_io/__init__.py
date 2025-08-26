from .cache import load_normal_from_cache, save_normal_cache_png  # 形状: (3,R,R) / (S,3,R,R)
from .stable_normal_predictor import create_predictor  # 形状: 工厂函数

__all__ = [
    "load_normal_from_cache",
    "save_normal_cache_png",
    "create_predictor",
]



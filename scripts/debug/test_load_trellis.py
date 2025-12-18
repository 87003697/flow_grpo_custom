import sys
import os
from pathlib import Path

# 添加 TRELLIS 路径
project_root = Path(".").resolve()
sys.path.insert(0, str(project_root / "_reference_codes" / "TRELLIS"))

from trellis.pipelines import TrellisImageTo3DPipeline

model_path = "pretrained_weights/TRELLIS-image-large"

try:
    print(f"Testing loading from {model_path}...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    print("✅ Successfully loaded pipeline!")
except Exception as e:
    print(f"❌ Failed to load pipeline: {e}")
    import traceback
    traceback.print_exc()

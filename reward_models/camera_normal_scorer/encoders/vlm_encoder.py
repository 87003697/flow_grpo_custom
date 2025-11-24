import base64
import io
import re
import time
from typing import List, Sequence

import requests
import torch
from PIL import Image

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_PROMPT = (
    "You are a vision critic. Compare the first normal map (reference) with the second "
    "normal map (candidate). Score alignment quality between 0 and 1. Output only "
    "\"Final Score: <float>\"."
)
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_BATCH_SIZE = 2
DEFAULT_SCORE_REGEX = r"Final Score:\s*([0-9]*\.?[0-9]+)"


class _VLMRequestError(RuntimeError):
    """Raised when the VLM endpoint fails after retries."""


class BaseVLMEncoder:
    """抽象 VLM 编码器：将两张图像送入多模态模型并返回匹配分数。"""

    def __init__(
        self,
        device: torch.device,
        score_min: float = 0.0,
        score_max: float = 1.0,
    ) -> None:
        self.device = device
        self.score_min = float(score_min)
        self.score_max = float(score_max)

    def score_pairs(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        mask_mesh_px: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _normalize_score(self, value: float) -> float:
        """将任意实数裁剪并映射到 [0, 1]。"""
        span = max(1e-6, self.score_max - self.score_min)
        clipped = min(self.score_max, max(self.score_min, value))
        return (clipped - self.score_min) / span

    @staticmethod
    def _pil_to_base64(img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GeminiSimilarityEncoder(BaseVLMEncoder):
    """使用 Gemini 2.5 (或兼容接口) 比较图像相似度。"""

    def __init__(
        self,
        device: torch.device,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        prompt_template: str | None = None,
        base_url: str | None = None,
        score_min: float = 0.0,
        score_max: float = 1.0,
    ) -> None:
        super().__init__(device=device, score_min=score_min, score_max=score_max)
        if len(api_key) == 0:
            raise ValueError("Gemini API key 未提供")
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.prompt_template = (prompt_template or DEFAULT_PROMPT).strip()
        self.timeout = DEFAULT_TIMEOUT
        self.max_retries = DEFAULT_MAX_RETRIES
        self.retry_delay = DEFAULT_RETRY_DELAY
        self.batch_size = DEFAULT_BATCH_SIZE
        self.score_pattern = re.compile(DEFAULT_SCORE_REGEX, flags=re.IGNORECASE)

    # === Gemini REST helpers ===
    def _build_url(self) -> str:
        return f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"

    def _build_payload(self, ref_b64: str, cand_b64: str) -> dict:
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": self.prompt_template,
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": ref_b64,
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": cand_b64,
                            }
                        },
                    ],
                }
            ],
        }

    def _extract_score(self, response_json: dict) -> float:
        text_parts: List[str] = []
        candidates = response_json.get("candidates") or []
        for cand in candidates:
            content = cand.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
        joined = "\n".join(text_parts)
        match = self.score_pattern.search(joined)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.0

    def _request_once(self, payload: dict) -> float:
        url = self._build_url()
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return self._extract_score(resp.json())
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)
        raise _VLMRequestError("Unexpected retry loop exit")

    def _score_pair(self, ref_img: Image.Image, cand_img: Image.Image) -> float:
        ref_b64 = self._pil_to_base64(ref_img)
        cand_b64 = self._pil_to_base64(cand_img)
        payload = self._build_payload(ref_b64, cand_b64)
        raw_score = self._request_once(payload)
        return self._normalize_score(raw_score)

    def score_pairs(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        mask_mesh_px: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        del mask_mesh_px  # Gemini 不使用掩码
        effective_bs = min(self.batch_size, max(1, int(batch_size)))
        total = len(mesh_pils)
        indices = list(mesh_group_indices)
        assert total == len(indices), "mesh_group_indices 长度需与样本一致"
        scores: List[float] = []
        for start in range(0, total, effective_bs):
            end = min(total, start + effective_bs)
            for j in range(start, end):
                gid = indices[j]
                ref_img = group_pils[gid]
                cand_img = mesh_pils[j]
                try:
                    score = self._score_pair(ref_img, cand_img)
                except Exception:
                    score = 0.0
                scores.append(score)
        score_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)  # 形状: (M,)
        return score_tensor

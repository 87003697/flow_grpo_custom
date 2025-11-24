import base64
import io
import re
from typing import List, Sequence

import requests
import torch
from PIL import Image


class GeminiSimilarityEncoder:
    """超轻量 Gemini VLM 打分器。"""

    SCORE_RE = re.compile(r"Final Score:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    PROMPT = (
        "You are a vision critic. Compare the first normal map (reference) with the second "
        "normal map (candidate). Score alignment quality between 0 and 1. Output only "
        "\"Final Score: <float>\"."
    )

    def __init__(self, device: torch.device, *, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("Gemini API key 未提供")
        self.device = device
        self.api_key = api_key
        self.model = model

    def _score_pair(self, ref_img: Image.Image, cand_img: Image.Image) -> float:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": self.PROMPT},
                        {"inline_data": {"mime_type": "image/png", "data": self._to_b64(ref_img)}},
                        {"inline_data": {"mime_type": "image/png", "data": self._to_b64(cand_img)}},
                    ],
                }
            ]
        }
        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        resp = requests.post(url, json=payload, timeout=30.0)
        resp.raise_for_status()
        text = "\n".join(
            part["text"]
            for cand in resp.json().get("candidates", [])
            for part in cand.get("content", {}).get("parts", [])
            if "text" in part
        )
        match = self.SCORE_RE.search(text)
        return float(match.group(1)) if match else 0.0

    def score_pairs(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        scores: List[float] = []
        total = len(mesh_pils)
        indices = list(mesh_group_indices)
        assert total == len(indices), "mesh_group_indices 长度需与样本一致"
        for j in range(total):
            gid = indices[j]
            scores.append(self._score_pair(group_pils[gid], mesh_pils[j]))
        return torch.tensor(scores, device=self.device, dtype=torch.float32)

    @staticmethod
    def _to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

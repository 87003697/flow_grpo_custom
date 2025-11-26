import asyncio
import base64
import io
import random
import re
from typing import List, Sequence

import aiohttp
import torch
from PIL import Image

PROMPT_TEMPLATES = {
    "v1": (
        "You are an expert 3D artist and mesh evaluator.\n"
        "You will be given two images. The first image is the reference image, and the second image is the rendered normal map of a 3D mesh reconstructed from the reference image.\n\n"
        "Please evaluate how similar the two normal maps are in terms of:\n"
        "- Consistency of edges, contours, and fine geometric structures.\n"
        "- Absence of artifacts such as blurry shapes, missing regions, or severe distortions.\n\n"
        "Your task is to evaluate how well the rendered normal map matches the reference image:\n"
        "- 1.0 means the reconstructed 3D mesh shows highly detailed geometric structures that are consistent with the reference image.\n"
        "- 0.5 means the reconstructed 3D mesh shows basic geometric structures that are roughly consistent with the reference image, but many fine details are missing or unclear.\n"
        "- 0.0 means the reconstructed 3D mesh does not show geometric structures that are consistent with the reference image.\n\n"
        "Reply with ONLY a number between 0.0 and 1.0, nothing else."
    ),
    "v2": (
        "You are an expert 3D artist and mesh evaluator.\n"
        "You will be given two images: the first is the reference image, and the second is the rendered normal map of a 3D mesh.\n\n"
        "Evaluate how well the mesh faithfully reconstructs the reference image, focusing ONLY on geometric structure (not color or texture).\n\n"
        "In your internal reasoning, consider:\n"
        "1. Identify the main subjects that the reference image is about, and its representative accessarys and parts that are crucial for the reconstruction.\n"
        "2. Evaluate how well the mesh accurately reconstructs the reference image and each of its parts, allowing small differences in camera viewpoint on the rendered normal map.\n"
        "3. Evaluate the absence of artifacts such as blurry shapes, holes, missing parts.\n\n"
        "4. Evaluate how well the contour, edges, convexities of each reprenstatitive part of the reconstructed mesh correspond to the reference image.\n`"
        "4. Evaluate the plausibility of the reconstructed 3D mesh, admitting the reconstructed mesh is semantically correct but visually slightly different from the reference image.\n\n"
        "Aggregate these into a single score:\n"
        "- 1.00: the shapes and geometric details are highly consistent with the reference image.\n"
        "- 0.00: the shapes and geometric details do not match the reference image.\n"
        "- Intermediate values: partially matched shapes and geometric details.\n\n"
        "Think step by step internally and keep all reasoning in your hidden thought process.\n"
        "In the final answer, do not reveal your reasoning; reply with ONLY a two decimal places number between 0.00 and 1.00."
    ),
}
API_KEYS = {
    "1": "sk-rQ9o21KZbQLcS6ssLMvqmBDUyHRHEXfKiPW5HpqwdilJqkR8",
    "2": "sk-ZrDsS3UAbUZyHMT9W4ZkftRZbHDN1FKrIx7QKl20bRcJISu1",
    "3": "sk-edfJqaOBuEbKfr7lM2w5Jt9p6J6Zfudokx1MK6cAbvgTf2MX",
}
BASE_URLS = {
    "1": "https://api5.xhub.chat/v1",
    "2": "https://api5.xhub.chat/v1",
    "3": "https://api5.xhub.chat/v1",
}

class GeminiOpenAIEncoder:
    """OpenAI 兼容格式的 Gemini VLM 打分器，支持高并发异步请求。"""

    SCORE_RE = re.compile(r"([0-9]*\.?[0-9]+)", re.IGNORECASE)

    def __init__(
        self,
        device: torch.device,
        *,
        api_source: str,
        model: str = "gemini-2.5-flash",
        max_concurrent: int = 8,
        timeout: float = 180.0,
        prompt_version: str = "v1",
        max_tokens: int = 200,
        thinking_enabled: bool = False,
        debug_raw_response: bool = False,
    ) -> None:
        # 根据 api_source 自动选择 API key 和 base_url（均为标量字符串）
        if api_source not in API_KEYS or api_source not in BASE_URLS:
            raise ValueError(f"未知的 api_source: {api_source}，必须是 '1'、'2' 或 '3'")
        api_key = API_KEYS[api_source]
        base_url = BASE_URLS[api_source]
        self.device = device
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = 4
        self.max_tokens = int(max_tokens)
        self.thinking_enabled = bool(thinking_enabled)
        self.debug_raw_response = debug_raw_response  # 形状: 布尔
        prompt = PROMPT_TEMPLATES.get(prompt_version)
        if prompt is None:
            raise ValueError(f"未知 prompt 版本: {prompt_version}")
        self.prompt = prompt

    async def _score_pair_async(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, ref_img: Image.Image, cand_img: Image.Image) -> float:
        """异步评分单对图像"""
        ref_b64 = self._to_b64(ref_img)  # 形状: 字符串
        cand_b64 = self._to_b64(cand_img)  # 形状: 字符串

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cand_b64}"}},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,  # 形状: 标量
            "temperature": 0.0,  # 形状: 标量
        }
        if self.thinking_enabled:
            payload["reasoning"] = {"effort": "low"}  # 形状: 字典

        for attempt in range(self.max_retries):
            try:
                async with semaphore:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()  # 形状: 字典
                        if self.debug_raw_response:
                            print("[GeminiOpenAIEncoder] raw response:", data)  # 形状: 字符串
                        text = data["choices"][0]["message"]["content"]  # 形状: 字符串
                        match = self.SCORE_RE.search(text)  # 形状: Match 或 None
                        return float(match.group(1)) if match else 0.0  # 形状: 标量
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt + 1 >= self.max_retries:
                    raise exc
                wait = (2 ** attempt) * random.uniform(0.8, 1.2)
                await asyncio.sleep(wait)

    def score_pairs(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        """批量评分，内部使用异步并发"""
        total = len(mesh_pils)  # 形状: 标量
        indices = list(mesh_group_indices)  # 形状: 列表(total)
        assert total == len(indices), "mesh_group_indices 长度需与样本一致"

        async def _batch_score():
            semaphore = asyncio.Semaphore(self.max_concurrent)  # 形状: 信号量
            async with aiohttp.ClientSession() as session:  # 形状: 会话
                tasks = [
                    self._score_pair_async(session, semaphore, group_pils[indices[j]], mesh_pils[j])
                    for j in range(total)
                ]  # 形状: 列表(total)
                return await asyncio.gather(*tasks)  # 形状: 列表(total)

        loop = asyncio.new_event_loop()  # 形状: 事件循环
        asyncio.set_event_loop(loop)
        scores = loop.run_until_complete(_batch_score())  # 形状: 列表(total)
        loop.close()
        return torch.tensor(scores, device=self.device, dtype=torch.float32)  # 形状: (total,)

    @staticmethod
    def _to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

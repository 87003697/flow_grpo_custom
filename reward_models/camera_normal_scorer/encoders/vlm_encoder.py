import asyncio
import base64
import io
import random
import re
from typing import List, Sequence

import aiohttp
import torch
from PIL import Image

API_KEYS = {
    "1": "sk-rQ9o21KZbQLcS6ssLMvqmBDUyHRHEXfKiPW5HpqwdilJqkR8",
    "2": "sk-ZrDsS3UAbUZyHMT9W4ZkftRZbHDN1FKrIx7QKl20bRcJISu1",
    "3": "sk-edfJqaOBuEbKfr7lM2w5Jt9p6J6Zfudokx1MK6cAbvgTf2MX",
    "4": "adPShZlcc3mPi8dl3LmcRCAJ@3695",
    "5": "hcTw2wQx9fOBb3llHMyLf9mt@3695"
}
BASE_URLS = {
    "1": "https://api5.xhub.chat/v1",
    "2": "https://api5.xhub.chat/v1",
    "3": "https://api5.xhub.chat/v1",
    "4": "http://v2.open.venus.oa.com/llmproxy",
    "5": "http://v2.open.venus.oa.com/llmproxy",
}

class GeminiOpenAIEncoder:
    """OpenAI 兼容格式的 Gemini VLM 打分器，支持高并发异步请求。"""

    SCORE_RE = re.compile(r"([0-9]*\.?[0-9]+)", re.IGNORECASE)


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
        prompt = self.PROMPT_TEMPLATES.get(prompt_version)
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


class GeminiOpenAIGroupEncoder:
    """一次请求内对同一 group 的多个候选进行评分。"""

    SCORE_RE = re.compile(r"([0-9]*\.?[0-9]+)", re.IGNORECASE)

    PROMPT_TEMPLATES = {
        "v1": (
            "You are an expert 3D artist and mesh quality inspector.\n"
            "You will receive one reference RGB image followed by {candidate_count} candidate normal maps, "
            "all belonging to the same scene.\n\n"
            "For each candidate i (in the exact order provided), judge how well its 3D geometry matches the reference image, "
            "considering only geometry (shapes, contours, convexities, concavities, part relations) and ignoring color/texture.\n"
            "Check that important structures exist, proportions are reasonable, and there are no severe artifacts or missing parts.\n\n"
            "Output {candidate_count} similarity scores between 0.00 and 1.00, each with exactly two decimal places, "
            "as a comma-separated list: s1, s2, ..., s{candidate_count}. Do not add explanations."
        ),
        "v2": (
            "You are an expert 3D artist and mesh evaluator.\n"
            "You will be given one reference image and {candidate_count} rendered normal maps of 3D meshes reconstructed from that reference image.\n\n"
            "Your task is to evaluate, for each candidate i (from 1 to {candidate_count}), how well the 3D mesh geometry in its normal map matches the geometry implied by the reference image, "
            "focusing ONLY on geometric structure (not color or texture).\n\n"
            "In your internal reasoning, for each candidate, consider:\n"
            "1. Identify the main subjects that the reference image is about, and the representative accessories and parts that are crucial for the reconstruction.\n"
            "2. Evaluate how well the mesh in candidate i accurately reconstructs the reference image and each of its important parts, allowing small differences in camera viewpoint on the rendered normal map.\n"
            "3. Evaluate the absence of artifacts such as blurry shapes, holes, missing parts, or severely distorted regions.\n"
            "4. Evaluate how well the contours, edges, and convexities of each representative part of the reconstructed mesh in candidate i correspond to the reference image.\n"
            "5. Evaluate the plausibility of the reconstructed 3D mesh in candidate i, admitting that it may be semantically correct but visually slightly different from the reference image.\n\n"
            "For each candidate i, aggregate these into a single similarity score between 0.00 and 1.00:\n"
            "- 1.00: the shapes and geometric details are highly consistent with the reference image.\n"
            "- 0.00: the shapes and geometric details do not match the reference image.\n"
            "- Intermediate values: partially matched shapes and geometric details.\n\n"
            "When there is only 1 candidate (i.e., {candidate_count} == 1), this score should be your best absolute estimate based solely on the criteria above.\n"
            "When there are multiple candidates (i.e., {candidate_count} > 1), you must still base each score on the same absolute criteria, "
            "but you should then use the relative differences between candidates to adjust the scores so that the final numeric values clearly encode which candidates are better or worse within the group.\n\n"
            "Think step by step internally and keep all reasoning in your hidden thought process.\n"
            "In the final answer, do not reveal your reasoning. Reply with ONLY {candidate_count} numbers between 0.00 and 1.00, "
            "each with exactly two decimal places, in order from candidate 1 to candidate {candidate_count}, "
            "separated by commas, for example: 0.87, 0.53, 0.22.\n"
        ),
    }

    def __init__(
        self,
        device: torch.device,
        *,
        api_source: str,
        model: str = "gemini-2.5-flash",
        max_concurrent: int = 4,
        timeout: float = 180.0,
        prompt_version: str = "v1",
        max_tokens: int = 512,
        thinking_enabled: bool = False,
        debug_raw_response: bool = False,
    ) -> None:
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
        prompt = self.PROMPT_TEMPLATES.get(prompt_version)
        if prompt is None:
            raise ValueError(f"未知 prompt 版本: {prompt_version}")
        self.prompt = prompt

    async def _score_group_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        ref_img: Image.Image,
        cand_imgs: List[Image.Image],
    ) -> List[float]:
        if not cand_imgs:
            return []

        ref_b64 = GeminiOpenAIEncoder._to_b64(ref_img)  # 形状: 字符串
        cand_b64_list = [GeminiOpenAIEncoder._to_b64(img) for img in cand_imgs]  # 形状: 列表(num_cand)
        prompt_text = self.prompt.format(candidate_count=len(cand_imgs))

        content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
        ]
        for idx, cand_b64 in enumerate(cand_b64_list, start=1):
            content.append({"type": "text", "text": f"Candidate #{idx}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cand_b64}"}})

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
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
                            print("[GeminiOpenAIGroupEncoder] raw response:", data)  # 形状: 字符串
                        choice = (data.get("choices") or [None])[0]  # 形状: 字典或None
                        text = (choice or {}).get("message", {}).get("content")  # 形状: 可选字符串
                        if not text:
                            continue
                        matches = self.SCORE_RE.findall(text)  # 形状: 列表(num_found)
                        scores = []
                        for value in matches[: len(cand_imgs)]:
                            try:
                                scores.append(float(value))
                            except ValueError:
                                scores.append(0.0)
                        if len(scores) < len(cand_imgs):
                            scores.extend([0.0] * (len(cand_imgs) - len(scores)))
                        return scores  # 形状: 列表(num_cand)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt + 1 >= self.max_retries:
                    raise exc
                wait = (2 ** attempt) * random.uniform(0.8, 1.2)
                await asyncio.sleep(wait)

        return [0.0] * len(cand_imgs)  # 形状: 列表(num_cand)

    def score_pairs(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        """将同组候选合并到单次请求进行评分。"""

        total = len(mesh_pils)  # 形状: 标量
        indices = list(mesh_group_indices)  # 形状: 列表(total)
        assert total == len(indices), "mesh_group_indices 长度需与样本一致"

        group_to_items = {}
        for mesh_idx, group_idx in enumerate(indices):
            group_to_items.setdefault(group_idx, []).append(mesh_idx)

        async def _batch_score():
            semaphore = asyncio.Semaphore(self.max_concurrent)  # 形状: 信号量
            async with aiohttp.ClientSession() as session:  # 形状: 会话
                grouped_entries = list(group_to_items.items())  # 形状: 列表(num_groups)
                tasks = [
                    self._score_group_async(
                        session,
                        semaphore,
                        group_pils[group_idx],
                        [mesh_pils[m_idx] for m_idx in mesh_indices],
                    )
                    for group_idx, mesh_indices in grouped_entries
                ]  # 形状: 列表(num_groups)
                results = await asyncio.gather(*tasks)  # 形状: 列表(num_groups)
                return grouped_entries, results  # 形状: 元组(列表, 列表)

        loop = asyncio.new_event_loop()  # 形状: 事件循环
        asyncio.set_event_loop(loop)
        grouped_entries, grouped_scores = loop.run_until_complete(_batch_score())  # 形状: (列表, 列表)
        loop.close()

        flat_scores = [0.0] * total  # 形状: 列表(total)
        for (group_idx, mesh_indices), scores in zip(grouped_entries, grouped_scores):
            if len(scores) != len(mesh_indices):
                # 长度不匹配时进行裁剪或补零，避免崩溃
                adjusted = (scores + [0.0] * len(mesh_indices))[: len(mesh_indices)]
            else:
                adjusted = scores
            for mesh_idx, score in zip(mesh_indices, adjusted):
                flat_scores[mesh_idx] = score

        return torch.tensor(flat_scores, device=self.device, dtype=torch.float32)  # 形状: (total,)

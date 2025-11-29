import asyncio
import base64
import io
import json
import os
import random
import time
from typing import Any, Awaitable, Callable, List, Optional, Sequence

import aiohttp
import requests
import torch
from PIL import Image

API_KEYS = {
    "1": [
        "sk-rQ9o21KZbQLcS6ssLMvqmBDUyHRHEXfKiPW5HpqwdilJqkR8",
    ],
    "2": [
        "sk-ZrDsS3UAbUZyHMT9W4ZkftRZbHDN1FKrIx7QKl20bRcJISu1",
    ],
    "3": [
        "sk-edfJqaOBuEbKfr7lM2w5Jt9p6J6Zfudokx1MK6cAbvgTf2MX",
    ],
    "4": [
        "adPShZlcc3mPi8dl3LmcRCAJ@3695",
        "hcTw2wQx9fOBb3llHMyLf9mt@3695",
    ],
    "5": [
        "NifjjUnlRK7h2l9oD63QqbVr@3695",
        "2DqbfqdmLrD2n9z9VDoGz4sE@3695",
    ],
}
BASE_URLS = {
    "1": "https://api5.xhub.chat/v1",
    "2": "https://api5.xhub.chat/v1",
    "3": "https://api5.xhub.chat/v1",
    "4": "http://v2.open.venus.oa.com/llmproxy",
    "5": "http://v2.open.venus.oa.com/llmproxy",
}

RETRY_STATUS_CODES = {401, 403, 408, 409, 429, 500, 502, 503, 504}


async def _retry_async(
    max_retries: int,
    action: Callable[[], Awaitable[Any]],
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return await action()
        except aiohttp.ClientResponseError as exc:
            last_error = exc
            if exc.status not in RETRY_STATUS_CODES:
                raise
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
        await asyncio.sleep((2 ** attempt) * random.uniform(0.8, 1.2))
    if last_error is not None:
        raise last_error
    raise RuntimeError("所有请求均重试失败（async）")


def _retry_sync(
    max_retries: int,
    action: Callable[[], Any],
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return action()
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status is None or status not in RETRY_STATUS_CODES:
                raise
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
        time.sleep((2 ** attempt) * random.uniform(0.8, 1.2))
    if last_error is not None:
        raise last_error
    raise RuntimeError("所有请求均重试失败（sync）")


class GeminiOpenAIEncoder:
    """OpenAI 兼容格式的 Gemini VLM 打分器，支持高并发异步请求。"""


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
            "You must reply strictly in JSON format using this template:\n"
            "{{\n"
            '  "score": <float_between_0_and_1>\n'
            "}}"
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
            "In the final answer, do not reveal your reasoning; respond strictly with JSON formatted exactly as:\n"
            "{{\n"
            '  "score": <float_between_0_and_1>\n'
            "}}"
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
        
        api_keys = list(API_KEYS[api_source])
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        selected_index = self.local_rank % len(api_keys)
        api_key = api_keys[selected_index]
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
            "response_format": {"type": "json_object"},  # 形状: 字典
        }
        if self.thinking_enabled:
            payload["reasoning"] = {"effort": "low"}  # 形状: 字典

        async def _request_once() -> float:
            async with semaphore:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    text_payload = await resp.text()  # 形状: 字符串
                    if resp.status >= 400:
                        print(
                            "[GeminiOpenAIEncoder] HTTP",
                            resp.status,
                            "resp:",
                            text_payload,
                        )  # 形状: 字符串
                        resp.raise_for_status()
                    data = json.loads(text_payload)  # 形状: 字典
                    if self.debug_raw_response:
                        print("[GeminiOpenAIEncoder] raw response:", data)  # 形状: 字符串
                    text = data["choices"][0]["message"]["content"]  # 形状: 字符串
                    parsed = json.loads(text)  # 形状: 字典
                    value = parsed.get("score", 0.0)  # 形状: 标量或字符串
                    return float(value)

        return await _retry_async(self.max_retries, _request_once)

    def score_pairs_async(
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

    PROMPT_TEMPLATES = {
        "v1": (
            "You are an expert 3D artist and mesh quality inspector.\n"
            "You will receive one reference RGB image followed by {candidate_count} candidate normal maps, "
            "all belonging to the same scene.\n\n"
            "For each candidate i (in the exact order provided), judge how well its 3D geometry matches the reference image, "
            "considering only geometry (shapes, contours, convexities, concavities, part relations) and ignoring color/texture.\n"
            "Check that important structures exist, proportions are reasonable, and there are no severe artifacts or missing parts.\n\n"
            "Output JSON only, following this example:\n"
            "{{\n"
            '  "scores": [\n'
            '    {{"candidate": 1, "score": "<score_candidate_1>"}},\n'
            '    {{"candidate": 2, "score": "<score_candidate_2>"}},\n'
            "    ...,\n"
            '    {{"candidate": {candidate_count}, "score": "<score_candidate_N>"}}\n'
            "  ]\n"
            "}}\n"
            "Use the same structure with {candidate_count} entries."
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
            "In the final answer, do not reveal your reasoning. Reply strictly with JSON using the schema shown below (update values accordingly):\n"
            "{{\n"
            '  "scores": [\n'
            '    {{"candidate": 1, "score": "<score_candidate_1>"}},\n'
            '    {{"candidate": 2, "score": "<score_candidate_2>"}},\n'
            "    ...,\n"
            '    {{"candidate": {candidate_count}, "score": "<score_candidate_N>"}}\n'
            "  ]\n"
            "}}\n"
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
        sync_mode: bool = True,
    ) -> None:
        if api_source not in API_KEYS or api_source not in BASE_URLS:
            raise ValueError(f"未知的 api_source: {api_source}，必须是 '1'、'2' 或 '3'")
        api_keys = list(API_KEYS[api_source])
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        selected_index = self.local_rank % len(api_keys)
        api_key = api_keys[selected_index]
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
        self.sync_mode = bool(sync_mode)
        prompt = self.PROMPT_TEMPLATES.get(prompt_version)

        self.prompt = prompt

    def _build_group_payload(self, ref_img: Image.Image, cand_imgs: List[Image.Image]) -> dict:
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
            "response_format": {"type": "json_object"},  # 形状: 字典
        }
        if self.thinking_enabled:
            payload["reasoning"] = {"effort": "low"}  # 形状: 字典
        return payload  # 形状: 字典

    async def _score_group_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        ref_img: Image.Image,
        cand_imgs: List[Image.Image],
    ) -> List[float]:
        if not cand_imgs:
            return []

        payload = self._build_group_payload(ref_img, cand_imgs)  # 形状: 字典

        async def _request_once() -> List[float]:
            async with semaphore:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    text_payload = await resp.text()  # 形状: 字符串
                    if resp.status >= 400:
                        print(
                            "[GeminiOpenAIGroupEncoder] HTTP",
                            resp.status,
                            "resp:",
                            text_payload,
                        )  # 形状: 字符串
                        resp.raise_for_status()
                    data = json.loads(text_payload)  # 形状: 字典
                    if self.debug_raw_response:
                        print("[GeminiOpenAIGroupEncoder] raw response:", data)  # 形状: 字符串
                    choice = (data.get("choices") or [None])[0]  # 形状: 字典或None
                    text = (choice or {}).get("message", {}).get("content")  # 形状: 可选字符串
                    if not text:
                        raise RuntimeError("VLM response missing content")
                    parsed = json.loads(text)  # 形状: 字典
                    items = parsed.get("scores")  # 形状: 可选列表
                    assert isinstance(items, list), "scores must be a list"
                    return [float(item.get("score", 0.0)) for item in items[: len(cand_imgs)]]

        return await _retry_async(self.max_retries, _request_once)

    def _score_group_sync(self, ref_img: Image.Image, cand_imgs: List[Image.Image]) -> List[float]:
        if not cand_imgs:
            return []

        payload = self._build_group_payload(ref_img, cand_imgs)  # 形状: 字典

        def _request_once() -> List[float]:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},  # 形状: 字典
                json=payload,
                timeout=self.timeout,
            )
            text_payload = resp.text  # 形状: 字符串
            if resp.status_code >= 400:
                print(
                    "[GeminiOpenAIGroupEncoder] HTTP",
                    resp.status_code,
                    "resp:",
                    text_payload,
                )  # 形状: 字符串
                resp.raise_for_status()
            data = json.loads(text_payload)  # 形状: 字典
            if self.debug_raw_response:
                print("[GeminiOpenAIGroupEncoder] raw response:", data)  # 形状: 字符串
            choice = (data.get("choices") or [None])[0]  # 形状: 字典或None
            text = (choice or {}).get("message", {}).get("content")  # 形状: 可选字符串
            if not text:
                raise RuntimeError("VLM response missing content")
            parsed = json.loads(text)  # 形状: 字典
            items = parsed.get("scores")  # 形状: 可选列表
            assert isinstance(items, list), "scores must be a list"
            return [float(item.get("score", 0.0)) for item in items[: len(cand_imgs)]]

        return _retry_sync(self.max_retries, _request_once)

    def score_pairs_async(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        """异步模式：同组候选合并到单次请求并并发执行。"""

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

        flat_scores = self._fill_scores(total, grouped_entries, grouped_scores)  # 形状: 列表(total)
        return torch.tensor(flat_scores, device=self.device, dtype=torch.float32)  # 形状: (total,)

    def score_pairs_sync(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        total = len(mesh_pils)  # 形状: 标量
        indices = list(mesh_group_indices)  # 形状: 列表(total)
        assert total == len(indices), "mesh_group_indices 长度需与样本一致"

        group_to_items = {}
        for mesh_idx, group_idx in enumerate(indices):
            group_to_items.setdefault(group_idx, []).append(mesh_idx)

        grouped_entries = list(group_to_items.items())  # 形状: 列表(num_groups)
        grouped_scores = [
            self._score_group_sync(
                group_pils[group_idx],
                [mesh_pils[m_idx] for m_idx in mesh_indices],
            )
            for group_idx, mesh_indices in grouped_entries
        ]  # 形状: 列表(num_groups)

        flat_scores = self._fill_scores(total, grouped_entries, grouped_scores)  # 形状: 列表(total)
        return torch.tensor(flat_scores, device=self.device, dtype=torch.float32)  # 形状: (total,)

    def _fill_scores(self, total, grouped_entries, grouped_scores):
        flat_scores = [0.0] * total  # 形状: 列表(total)
        for (_, mesh_indices), scores in zip(grouped_entries, grouped_scores):
            if len(scores) != len(mesh_indices):
                raise ValueError("VLM 返回数量与候选数不符")  # 形状: 异常
            for mesh_idx, score in zip(mesh_indices, scores):
                flat_scores[mesh_idx] = float(score)
        return flat_scores  # 形状: 列表(total)

    def score_pairs(
        self,
        group_pils: List[Image.Image],
        mesh_pils: List[Image.Image],
        mesh_group_indices: Sequence[int],
        **kwargs,
    ) -> torch.Tensor:
        if self.sync_mode:
            return self.score_pairs_sync(group_pils, mesh_pils, mesh_group_indices, **kwargs)
        else:
            return self.score_pairs_async(group_pils, mesh_pils, mesh_group_indices, **kwargs)

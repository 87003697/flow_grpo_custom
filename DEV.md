# Camera Normal Scorer VLM 验证流程

## 目标
- 复用 `dataset/meshes_benchmark_v1` 提供的 `(image, normal, meshes[])` 记录。
- 调用 `reward_models.camera_normal_scorer.CameraNormalScorer`，针对同一 image 下的多个 mesh 生成 VLM/Gemini 分数。
- 将参考法线、渲染法线与得分保存下来，便于人工验证。

## 数据装载
1. 读取 `dataset_index.json`，每条记录包含：
   - `image`: RGB 输入路径。
   - `normal`: 图像侧法线路径。
   - `meshes`: 多个 `{path, pipeline}` 候选。
2. 固定 image/normal 后，遍历 `meshes` 列表：
   - 使用 `trimesh.load(path, process=False)` 加载，再转成 `MeshExtractResult`。
   - 将同一个 `Image.open(image_path).convert("RGB")` 与 `Image.open(normal_path).convert("RGB")` 复制给每个 mesh 对应的 `images` 与 `metadata` 条目，保证 `metadata[i] = {"image_path": image_path, "normal_pil": shared_normal_pil}`。

## 评分流程
1. 构造 `CameraNormalScorer`：
   ```python
   from reward_models.camera_normal_scorer.scorer import CameraNormalScorer

   cfg = {
       "normal_resolution": 512,
       "cache_dir": "./tmp_camera_normal_cache",
       "encoder": "gemini-2.5-flash",  # 或 "gemini-2.5-flash-group"
       "camera_ckpt": "/path/to/vggt_checkpoint.pt",
       "camera_type": "search",
       "vlm_api_source": "1",
       "vlm_max_concurrent": 2,
       "vlm_timeout": 180.0,
       "vlm_prompt_version": "v1",
   }
   scorer = CameraNormalScorer(device=torch.device("cuda"), cfg=cfg)
   ```
2. 调用：
   ```python
   scores, grouped_meta = scorer.compute_scores(meshes, images, metadata)
   ```
   - `scores` 与输入 mesh 顺序对齐。
   - `grouped_meta` 记录每张 image 的组信息以及所有候选的渲染法线、得分。

## 可复用模块来源
来自 `scripts/eval/eval_mesh_scorer_eval3d.py`，可直接拷贝或适配到新的验证脚本：

- `load_glb_mesh_as_obj()`：统一读取 `.glb/.ply/.obj` 并返回带 `vertices`/`faces` 的简易对象。
- `_cache_path_from_image()` / `load_normal_pil_from_cache()`：按 image 名从缓存目录获取法线 PNG，避免重复推理。
- `_rotate_meshes_by_source_front()`：根据 `source_front` 参数将多份 mesh 对齐至统一坐标系。
- `cfg` 字典构造示例：展示如何把 CLI 参数映射到 `CameraNormalScorer` 所需字段（`normal_resolution`、`cache_dir`、`encoder`、`camera_config_py`、`camera_ckpt` 等）。
- CSV 输出骨架：批量评分后写入 `[name, image, mesh, score]` 并附带平均分，便于快速统计。

## 可视化与落盘
1. 生成“多 mesh 对比”法线可视化（参考法线 + 所有候选渲染 + 分数）：
   ```python
   from pathlib import Path
    from math import ceil
    from PIL import Image, ImageDraw, ImageFont

    for grp in grouped_meta:
        # 为每张 image 单独创建子目录，便于逐组查看
        base = Path(save_dir) / Path(grp["image_path"]).stem
        base.mkdir(parents=True, exist_ok=True)

        cands_sorted = sorted(
            grp["candidates"],
            key=lambda c: c["score"] if c["score"] is not None else -1.0,
            reverse=True,
        )
        tiles = [{"label": "reference", "score": None, "pil": grp["image_normal_pil"]}]
        for cand in cands_sorted:
            tiles.append({
                "label": f"mesh_{cand['mesh_index']}",
                "score": cand["score"],
                "pil": cand["rendered_normal_pil"],
            })

        cols = 4
        rows = ceil(len(tiles) / cols)
        w, h = tiles[0]["pil"].size
        canvas = Image.new("RGB", (cols * w, rows * h), "black")
        draw = ImageDraw.Draw(canvas)
        font = None  # 使用默认字体

        for idx, tile in enumerate(tiles):
            r, c = divmod(idx, cols)
            x0, y0 = c * w, r * h
            canvas.paste(tile["pil"], (x0, y0))
            label = tile["label"]
            if tile["score"] is not None:
                label += f" | {tile['score']:.3f}"
            draw.rectangle([x0, y0, x0 + w, y0 + 22], fill=(0, 0, 0, 180))
            draw.text((x0 + 4, y0 + 4), label, fill=(255, 255, 255), font=font)

        canvas.save(base / "normal_comparison.png")
   ```
2. 可选：将 `scores` 写入 CSV，并统计每个 pipeline 的均值/方差。
3. 若需要更完整的可视化，可在 `ScorerConfig` 中设置 `save_vis=True`，内部会调用 `save_camera_search_visualization` 输出相机搜索过程。

## 建议运行命令
```bash
python scripts/verify_vlm_meshes_benchmark.py \
  --dataset-root dataset/meshes_benchmark_v1 \
  --camera-ckpt /path/to/vggt_checkpoint.pt \
  --encoder gemini-2.5-flash \
  --max-count 10 \
  --save-dir outputs/vlm_vis_demo
```
- 该脚本需自行实现（参考 `_reference_codes/VGGTObj/scripts/meshes_benchmark_infer.py` 结构），核心逻辑与上文一致。
- 建议先用 `--max-count` 做抽检，再跑全量并统计得分分布。

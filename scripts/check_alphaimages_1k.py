import os
import json
import glob


def list_files(directory_path):
    allowed_exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [
        file_path
        for file_path in glob.glob(os.path.join(directory_path, "*"))
        if os.path.isfile(file_path)
        and os.path.splitext(file_path)[1].lower() in allowed_exts
    ]


def _check_header_signature(file_path):
    """Best-effort快速签名校验，返回(None)表示通过，否则返回错误原因。"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)
    except Exception as error:
        return f"read_failed: {error}"

    # PNG
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    # JPEG
    if header.startswith(b"\xff\xd8"):
        return None
    # WEBP (RIFF....WEBP)
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return None
    # 兜底：无法识别或签名异常
    return "unknown_or_bad_signature"


def is_valid_image(file_path):
    """返回 (ok: bool, error: Optional[str])"""
    try:
        size_bytes = os.path.getsize(file_path)
    except Exception as error:
        return False, f"stat_failed: {error}"

    if size_bytes == 0:
        return False, "zero_size"

    sig_error = _check_header_signature(file_path)
    if sig_error is not None:
        return False, sig_error

    return True, None


def scan_split(root_dir, split_name):
    images_dir = os.path.join(root_dir, split_name, "images")
    normals_dir = os.path.join(root_dir, split_name, "normals")

    results = {
        "bad_images": [],
        "zero_size_images": [],
        "bad_normals": [],
        "zero_size_normals": [],
        "missing_normals": [],
        "orphan_normals": [],
    }

    if not os.path.isdir(images_dir) or not os.path.isdir(normals_dir):
        return results

    image_files = list_files(images_dir)
    normal_files = list_files(normals_dir)

    image_name_to_path = {
        os.path.splitext(os.path.basename(file_path))[0]: file_path
        for file_path in image_files
    }
    normal_name_to_path = {
        os.path.splitext(os.path.basename(file_path))[0]: file_path
        for file_path in normal_files
    }

    for base_name, file_path in image_name_to_path.items():
        ok, err = is_valid_image(file_path)
        if not ok:
            if err == "zero_size":
                results["zero_size_images"].append(file_path)
            else:
                results["bad_images"].append({
                    "path": file_path,
                    "error": err,
                })

    for base_name, file_path in normal_name_to_path.items():
        ok, err = is_valid_image(file_path)
        if not ok:
            if err == "zero_size":
                results["zero_size_normals"].append(file_path)
            else:
                results["bad_normals"].append({
                    "path": file_path,
                    "error": err,
                })

    for base_name, file_path in image_name_to_path.items():
        if base_name not in normal_name_to_path:
            results["missing_normals"].append({
                "image": file_path,
                "expected_normal": os.path.join(normals_dir, base_name + ".png"),
            })

    for base_name, file_path in normal_name_to_path.items():
        if base_name not in image_name_to_path:
            results["orphan_normals"].append(file_path)

    return results


def main():
    dataset_root = "/data/zhiyuan_ma/code/flow_grpo_custom/dataset/alphaimages_1k"

    report = {
        "train": scan_split(dataset_root, "train"),
        "test": scan_split(dataset_root, "test"),
    }

    output_path = os.path.join(dataset_root, "alphaimages_1k_scan.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print(output_path)


if __name__ == "__main__":
    main()



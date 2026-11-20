from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml


def list_images(images_dir: Path, extensions: Sequence[str]) -> list[Path]:
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def resolve_mask_path(
    image_path: Path,
    masks_split_dir: Path,
    mask_suffix: str | None = None,
    mask_extension: str | None = None,
) -> Path:
    if mask_suffix:
        return masks_split_dir / f"{image_path.stem}{mask_suffix}"
    if mask_extension:
        ext = mask_extension if mask_extension.startswith(".") else f".{mask_extension}"
        return masks_split_dir / f"{image_path.stem}{ext}"
    return masks_split_dir / image_path.name


def mask_to_yolo_polygons(
    mask: np.ndarray,
    road_class_id: int,
    min_area_px: float = 0.0,
    approx_epsilon_ratio: float = 0.0,
) -> list[str]:
    """Convert a 2D semantic mask into YOLO segmentation label lines for a single class (id 0)."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {mask.shape}")

    height, width = mask.shape
    if height == 0 or width == 0:
        raise ValueError(f"Invalid mask dimensions: {(height, width)}")

    binary_mask = (mask == road_class_id).astype(np.uint8)
    if not np.any(binary_mask):
        return []

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines: list[str] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_px:
            continue

        pts = contour
        if approx_epsilon_ratio > 0.0:
            perimeter = cv2.arcLength(contour, closed=True)
            epsilon = approx_epsilon_ratio * perimeter
            pts = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)

        # pts has shape (N, 1, 2)
        coords = pts.reshape(-1, 2).astype(np.float64)
        if len(coords) < 3:
            continue

        coords[:, 0] = np.clip(coords[:, 0] / float(width), 0.0, 1.0)
        coords[:, 1] = np.clip(coords[:, 1] / float(height), 0.0, 1.0)

        # Remove consecutive duplicate normalized points if simplification collapsed them
        deduped = [coords[0]]
        for pt in coords[1:]:
            if not np.allclose(pt, deduped[-1], atol=1e-7):
                deduped.append(pt)
        if len(deduped) > 1 and np.allclose(deduped[-1], deduped[0], atol=1e-7):
            deduped.pop()

        if len(deduped) < 3:
            continue

        flat_coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in deduped)
        yolo_lines.append(f"0 {flat_coords}")

    return yolo_lines


def place_image(src: Path, dst: Path, link_images: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_images:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def convert_split(
    split: str,
    raw_images_dir: Path,
    raw_masks_dir: Path,
    output_dir: Path,
    road_class_id: int,
    image_extensions: Sequence[str],
    mask_suffix: str | None = None,
    mask_extension: str | None = None,
    min_area_px: float = 0.0,
    approx_epsilon_ratio: float = 0.0,
    link_images: bool = False,
) -> dict[str, int]:
    img_split_dir = raw_images_dir / split
    mask_split_dir = raw_masks_dir / split

    if not img_split_dir.exists():
        raise FileNotFoundError(f"Missing raw images split directory: {img_split_dir}")
    if not mask_split_dir.exists():
        raise FileNotFoundError(f"Missing raw masks split directory: {mask_split_dir}")

    out_img_dir = output_dir / "images" / split
    out_lbl_dir = output_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(img_split_dir, image_extensions)
    stats = {
        "total_images": len(image_paths),
        "positive_images": 0,
        "empty_label_images": 0,
        "total_polygons": 0,
    }

    for img_path in image_paths:
        mask_path = resolve_mask_path(
            image_path=img_path,
            masks_split_dir=mask_split_dir,
            mask_suffix=mask_suffix,
            mask_extension=mask_extension,
        )
        if not mask_path.exists():
            raise FileNotFoundError(f"Corresponding mask not found for {img_path}: {mask_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img_h, img_w = img.shape[:2]

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if mask.shape != (img_h, img_w):
            raise ValueError(
                f"Dimension mismatch for {img_path.name}: "
                f"image is {(img_h, img_w)}, mask is {mask.shape}"
            )

        yolo_lines = mask_to_yolo_polygons(
            mask=mask,
            road_class_id=road_class_id,
            min_area_px=min_area_px,
            approx_epsilon_ratio=approx_epsilon_ratio,
        )

        place_image(img_path, out_img_dir / img_path.name, link_images=link_images)

        lbl_path = out_lbl_dir / f"{img_path.stem}.txt"
        content = "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
        lbl_path.write_text(content, encoding="utf-8")

        if yolo_lines:
            stats["positive_images"] += 1
            stats["total_polygons"] += len(yolo_lines)
        else:
            stats["empty_label_images"] += 1

    return stats


def write_dataset_yaml(output_dir: Path, splits: Iterable[str] = ("train", "val")) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "names": {0: "road"},
    }
    for split in splits:
        data[split] = f"images/{split}"

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path
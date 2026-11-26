from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.utils import DEFAULT_CFG


def create_segment_dataloader(
    dataset_yaml: Path,
    split: str,
    batch_size: int = 8,
    imgsz: int = 640,
    workers: int = 2,
):
    """Build an Ultralytics segmentation dataset and dataloader for a given split ('train' or 'val')."""
    overrides = {
        "model": "yolo11n-seg.pt",
        "data": str(dataset_yaml),
        "imgsz": imgsz,
        "batch": batch_size,
        "workers": workers,
        "task": "segment",
    }
    cfg = get_cfg(DEFAULT_CFG, overrides)
    img_path = str(dataset_yaml.parent / "images" / split)

    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=img_path,
        batch=batch_size,
        data={"names": {0: "road"}, "nc": 1},
        mode="train" if split == "train" else "val",
        rect=False,
        stride=32,
    )
    dataloader = build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=False, rank=-1)
    return dataloader

def _denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert Ultralytics batch image tensor (C, H, W) [0..1] or [0..255] to uint8 BGR for OpenCV."""
    arr = img_tensor.detach().cpu().numpy()
    if arr.max() <= 1.0 + 1e-5:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Ultralytics images are RGB; convert to BGR for cv2 drawing/saving
    rgb = np.transpose(arr, (1, 2, 0))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_batch_overlays(
    batch: dict[str, Any],
    save_dir: Path,
    prefix: str,
) -> list[Path]:
    """Render ground-truth segmentation overlays and save images."""

    save_dir.mkdir(parents=True, exist_ok=True)

    imgs = batch["img"]
    batch_idx = batch.get("batch_idx")
    cls = batch.get("cls")
    masks = batch.get("masks")

    saved_paths: list[Path] = []
    num_items = imgs.shape[0]

    # Move indexing information to CPU
    if batch_idx is not None:
        batch_idx = batch_idx.detach().cpu().reshape(-1)

    if cls is not None:
        cls = cls.detach().cpu().reshape(-1)

    num_instances = len(batch_idx) if batch_idx is not None else 0

    for i in range(num_items):
        canvas = _denormalize_image(imgs[i])
        h, w = canvas.shape[:2]

        # Count polygons belonging to the current image
        sample_cls_count = 0

        if batch_idx is not None and cls is not None:
            sample_cls_count = int((batch_idx == i).sum().item())

        if masks is not None:
            masks = masks.detach().cpu()

            if masks.ndim != 3:
                raise ValueError(
                    f"Expected masks with 3 dimensions, got {masks.shape}"
                )

            # Case 1: Overlap masks - one mask per image
            if masks.shape[0] == num_items and masks.shape[0] != num_instances:
                combined = (masks[i] > 0).numpy().astype(np.uint8)

            # Case 2: Individual masks - one mask per instance
            elif masks.shape[0] == num_instances:
                if batch_idx is not None:
                    sel = batch_idx == i
                    sample_masks = masks[sel]

                    if sample_masks.shape[0] > 0:
                        combined = sample_masks.gt(0).any(dim=0).numpy().astype(np.uint8)
                    else:
                        combined = np.zeros(masks.shape[1:], dtype=np.uint8)
                else:
                    combined = np.zeros(masks.shape[1:], dtype=np.uint8)

            else:
                raise ValueError(
                    f"Cannot associate masks with images. "
                    f"Mask count: {masks.shape[0]}, "
                    f"batch size: {num_items}, "
                    f"instance count: {num_instances}"
                )

            # Resize mask to original batch image dimensions
            if combined.shape != (h, w):
                combined = cv2.resize(
                    combined,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Draw road overlay
            if combined.any():
                overlay = canvas.copy()
                overlay[combined > 0] = (0, 200, 0)

                canvas = cv2.addWeighted(
                    overlay, 0.4, canvas, 0.6, 0
                )

                contours, _ = cv2.findContours(
                    combined,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                cv2.drawContours(
                    canvas,
                    contours,
                    -1,
                    (0, 255, 0),
                    2,
                )

        # Add image information
        label_text = f"{prefix} | idx={i} | road_polys={sample_cls_count}"

        cv2.putText(
            canvas,
            label_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Save visualization
        out_path = save_dir / f"{prefix}_sample_{i:02d}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)

    return saved_paths

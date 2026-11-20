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
    """Render polygon/mask overlays from a YOLO dataloader batch and save images."""
    save_dir.mkdir(parents=True, exist_ok=True)
    imgs = batch["img"]
    batch_idx = batch.get("batch_idx")
    cls = batch.get("cls")
    masks = batch.get("masks")  # Present when polygon/mask targets are collated by Ultralytics

    saved_paths: list[Path] = []
    num_items = imgs.shape[0]

    for i in range(num_items):
        canvas = _denormalize_image(imgs[i])
        h, w = canvas.shape[:2]

        sample_cls_count = 0
        if batch_idx is not None and cls is not None:
            sample_mask = (batch_idx == i).detach().cpu().numpy().reshape(-1)
            sample_cls = cls.detach().cpu().numpy().reshape(-1)[sample_mask]
            sample_cls_count = int(len(sample_cls))

        if masks is not None:
            # masks shape in Ultralytics batches is typically (N_instances, H, W)
            m_batch_idx = batch.get("batch_idx")
            if m_batch_idx is not None:
                sel = (m_batch_idx == i).detach().cpu().numpy().reshape(-1)
                sample_masks = masks[sel].detach().cpu().numpy()
                if sample_masks.size > 0:
                    combined = np.clip(np.sum(sample_masks, axis=0), 0, 1).astype(np.uint8)
                    if combined.shape != (h, w):
                        combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)
                    overlay = canvas.copy()
                    overlay[combined > 0] = (0, 200, 0)  # green road tint
                    canvas = cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0)

                    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

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

        out_path = save_dir / f"{prefix}_sample_{i:02d}.png"
        cv2.imwrite(str(out_path), canvas)
        saved_paths.append(out_path)

    return saved_paths
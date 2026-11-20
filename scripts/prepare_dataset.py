from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mask_to_yolo import convert_split, write_dataset_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert 19-class road masks to YOLO segmentation format.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    poly_cfg = cfg["polygon"]

    output_dir = Path(ds_cfg["output_dir"])
    raw_images_dir = Path(ds_cfg["raw_images_dir"])
    raw_masks_dir = Path(ds_cfg["raw_masks_dir"])

    for split in ("train", "val"):
        stats = convert_split(
            split=split,
            raw_images_dir=raw_images_dir,
            raw_masks_dir=raw_masks_dir,
            output_dir=output_dir,
            road_class_id=int(poly_cfg["road_class_id"]),
            image_extensions=ds_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]),
            mask_suffix=ds_cfg.get("mask_suffix"),
            mask_extension=ds_cfg.get("mask_extension"),
            min_area_px=float(poly_cfg.get("min_area_px", 0.0)),
            approx_epsilon_ratio=float(poly_cfg.get("approx_epsilon_ratio", 0.0)),
            link_images=bool(ds_cfg.get("link_images", False)),
        )
        print(f"[{split}] {stats}")

    yaml_path = write_dataset_yaml(output_dir, splits=("train", "val"))
    print(f"Wrote dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
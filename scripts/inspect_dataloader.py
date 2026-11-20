from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualize_loader import create_segment_dataloader, render_batch_overlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO segmentation dataloader batches.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["dataset"]["output_dir"])
    dataset_yaml = output_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {dataset_yaml}. Run prepare_dataset.py first.")

    dl_cfg = cfg["dataloader"]
    save_root = Path(dl_cfg.get("save_dir", "runs/dataloader_vis"))

    for split in dl_cfg.get("splits_to_inspect", ["train", "val"]):
        loader = create_segment_dataloader(
            dataset_yaml=dataset_yaml,
            split=split,
            batch_size=int(dl_cfg.get("batch_size", 8)),
            imgsz=int(dl_cfg.get("imgsz", 640)),
            workers=int(dl_cfg.get("workers", 2)),
        )

        max_batches = int(dl_cfg.get("max_batches", 2))
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            img_shape = tuple(batch["img"].shape)
            cls_vals = batch["cls"].detach().cpu().numpy().tolist() if "cls" in batch else []
            print(f"[{split} | batch {batch_idx}] img={img_shape}, cls={cls_vals}")

            render_batch_overlays(
                batch=batch,
                save_dir=save_root / split,
                prefix=f"{split}_b{batch_idx:02d}",
            )


if __name__ == "__main__":
    main()
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from utils.yolo_viz import load_yolo_dataset_yaml, infer_labels_dir_from_images_dir, find_images


def _count_instances_in_labels_dir(labels_dir: Path) -> int:
    """Count YOLO instances as number of non-empty lines across all .txt files."""
    if not labels_dir.exists():
        return 0

    total = 0
    for txt in labels_dir.rglob("*.txt"):
        with txt.open("r") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def count_split(images_dir: Union[str, Path]) -> Tuple[int, int]:
    """
    Returns (n_images, n_instances) for a YOLO split given the images/<split> dir.
    Labels dir is inferred as labels/<split>.
    """
    images_dir = Path(images_dir)
    labels_dir = infer_labels_dir_from_images_dir(images_dir)

    n_images = len(find_images(images_dir))
    n_instances = _count_instances_in_labels_dir(labels_dir)
    return n_images, n_instances


def count_from_yaml(yaml_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    Returns dict like:
      {
        "train": {"images": ..., "instances": ...},
        "val":   {"images": ..., "instances": ...},
        "test":  {"images": ..., "instances": ...},
        "total": {"images": ..., "instances": ...},
      }
    """
    ds = load_yolo_dataset_yaml(yaml_path)

    split_dirs = {
        "train": ds.train,
        "val": ds.val,
        "test": ds.test,
    }

    out: Dict[str, Dict[str, int]] = {}
    total_images = 0
    total_instances = 0

    for split, img_dir in split_dirs.items():
        if img_dir is None:
            continue
        n_images, n_instances = count_split(img_dir)
        out[split] = {"images": n_images, "instances": n_instances}
        total_images += n_images
        total_instances += n_instances

    out["total"] = {"images": total_images, "instances": total_instances}
    return out

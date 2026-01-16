# utils/yolo_viz.py
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt

try:
    import yaml  # pip install pyyaml
except ImportError as e:
    raise ImportError("Missing dependency: pyyaml. Install with: pip install pyyaml") from e


@dataclass(frozen=True)
class YoloDataset:
    train: Optional[Path]
    val: Optional[Path]
    test: Optional[Path]
    names: List[str]


def load_yolo_dataset_yaml(yaml_path: Union[str, Path]) -> YoloDataset:
    """Load a YOLO dataset YAML with keys like train/val/test and names."""
    yaml_path = Path(yaml_path).expanduser().resolve()
    data = yaml.safe_load(yaml_path.read_text())

    def to_path(v) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser()

    names = data.get("names", [])
    if not isinstance(names, list) or not names:
        raise ValueError(f"'names' not found or invalid in {yaml_path}")

    return YoloDataset(
        train=to_path(data.get("train")),
        val=to_path(data.get("val")),
        test=to_path(data.get("test")),
        names=[str(x) for x in names],
    )


def find_images(folder: Path, exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[Path]:
    folder = Path(folder)
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def infer_labels_dir_from_images_dir(images_split_dir: Path) -> Path:
    """
    Given:  .../processed/images/train
    Return: .../processed/labels/train
    """
    images_split_dir = Path(images_split_dir)
    split = images_split_dir.name
    images_root = images_split_dir.parent          # .../processed/images
    processed_root = images_root.parent            # .../processed
    labels_split_dir = processed_root / "labels" / split
    return labels_split_dir


def label_path_for_image(img_path: Path, images_split_dir: Path, labels_split_dir: Path) -> Path:
    """
    Supports nested folders by preserving relative structure under images_split_dir.
    """
    img_path = Path(img_path)
    images_split_dir = Path(images_split_dir)
    labels_split_dir = Path(labels_split_dir)

    rel = img_path.relative_to(images_split_dir)
    return (labels_split_dir / rel).with_suffix(".txt")


def draw_yolo_labels_rgb(image_bgr, label_file: Path, names: Sequence[str]):
    """
    Draw YOLO labels (class cx cy w h) normalized [0..1] onto image.
    Returns RGB image for matplotlib.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]

    if not label_file.exists():
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for line in label_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = names[cls] if 0 <= cls < len(names) else f"id{cls}"
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_random_labeled_images(
    images_split_dir: Union[str, Path],
    names: Sequence[str],
    labels_split_dir: Optional[Union[str, Path]] = None,
    num_pics: int = 8,
    cols: int = 4,
    require_labels: bool = True,
    seed: Optional[int] = None,
    title_prefix: str = "",
):
    """
    Samples images and plots them with YOLO boxes.
    If labels_split_dir is None, it is inferred from images_split_dir.
    """
    images_split_dir = Path(images_split_dir)
    labels_split_dir = Path(labels_split_dir) if labels_split_dir else infer_labels_dir_from_images_dir(images_split_dir)

    all_imgs = find_images(images_split_dir)

    if require_labels:
        pool = [p for p in all_imgs if label_path_for_image(p, images_split_dir, labels_split_dir).exists()]
    else:
        pool = all_imgs

    if not pool:
        raise FileNotFoundError(f"No images found in {images_split_dir} (require_labels={require_labels})")

    if seed is not None:
        random.seed(seed)

    sample = random.sample(pool, min(num_pics, len(pool)))
    rows = math.ceil(len(sample) / cols)

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, img_path in enumerate(sample, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        lab = label_path_for_image(img_path, images_split_dir, labels_split_dir)
        vis = draw_yolo_labels_rgb(img_bgr, lab, names)

        plt.subplot(rows, cols, i)
        plt.imshow(vis)
        plt.axis("off")
        plt.title(f"{title_prefix}{img_path.name}")

    plt.tight_layout()
    plt.show()

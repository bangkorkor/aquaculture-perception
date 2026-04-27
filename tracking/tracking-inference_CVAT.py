#!/usr/bin/env python3
# THIS SCRIPT IS FOR TRACKING AND GETTING OUTPUT FOR CVAT, MEANING FRAMES ARE LABELED 1, 2, 3, til end
"""
RT-DETR + ByteTrack inference → MOT11 annotation export
=========================================================
Runs a trained RT-DETR model over a folder of images, tracks detections with
ByteTrack, and writes results in MOT11 format (.txt + labels + CVAT zip).

MOT11 row format:
    frame_id, track_id, x, y, w, h, not_ignored, class_id, visibility
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = (
    "../runs/detect/outputs/training/solaqua_fish"
    "/rt_detr_solaqua_fish_120e_fair/weights/best.pt"
)

SOURCE = (
    "../data-processing/vision/SOLAQUA/raw_processed"
    "/all_images/2024-08-20_17-14-36"
)

OUTPUT_DIR = Path("outputs/inference_annotation_MOT11_CVAT")

# Detector settings
CONF_THRESHOLD = 0.25
# RT-DETR handles duplicate suppression internally via its transformer decoder,
# so NMS is largely redundant. We set iou very high to avoid incorrectly
# discarding valid detections.
IOU_THRESHOLD  = 0.99

DEVICE         = "cuda:0"   # e.g. "cuda:0" for GPU
FRAME_RATE     = 30      # used by ByteTrack's Kalman filter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

LOG_INTERVAL = 50   # print progress every N frames


# =============================================================================
# HELPERS
# =============================================================================

def unique_path(path: Path) -> Path:
    """
    Return *path* if it does not exist, otherwise append _1, _2, … until a
    free filename is found.

    Examples
    --------
    >>> unique_path(Path("out/foo.txt"))   # "out/foo.txt" exists
    Path("out/foo_1.txt")
    """
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def numeric_sort_key(p: Path) -> tuple[int, int | str]:
    """Sort paths whose stems are integers numerically; others go last."""
    try:
        return (0, int(p.stem))
    except ValueError:
        return (1, p.stem)


def collect_images(folder: Path) -> list[Path]:
    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise RuntimeError(f"No images found in: {folder}")
    paths.sort(key=numeric_sort_key)
    return paths


def build_tracker(frame_rate: int) -> BYTETracker:
    cfg_path = check_yaml("../ultralytics/cfg/trackers/bytetrack.yaml")
    cfg = IterableSimpleNamespace(**YAML.load(cfg_path))
    return BYTETracker(args=cfg, frame_rate=frame_rate)


def detections_to_tracker_input(boxes) -> np.ndarray:
    """
    Convert an Ultralytics ``Boxes`` object to the array ByteTrack expects:
        shape (N, 6) → [x1, y1, x2, y2, confidence, class_id]

    We build this explicitly rather than relying on the undocumented internal
    layout of ``Boxes.data``.
    """
    return np.column_stack([
        boxes.xyxy.numpy(),   # (N, 4)
        boxes.conf.numpy(),   # (N,)
        boxes.cls.numpy(),    # (N,)
    ]).astype(np.float32)


def parse_tracks(
    tracks: np.ndarray,
    tracker_input: np.ndarray,
) -> list[tuple[int, int, np.ndarray, float]]:
    """
    Parse ByteTrack output rows into (track_id, class_id, xyxy, confidence).

    ByteTrack returns rows of varying column counts across Ultralytics versions.
    We handle the layout defensively:

    Known layouts
    -------------
    7 cols : [x1, y1, x2, y2, track_id, score, cls]
    8 cols : [x1, y1, x2, y2, track_id, score, cls, det_idx]

    When ``det_idx`` is present we use it to recover the **original detector
    box** (more precise than the Kalman-smoothed tracker box).  When it is
    absent we fall back to the tracker's own box.

    Returns
    -------
    List of (track_id, class_id, xyxy_array, confidence) tuples.
    """
    n_cols = tracks.shape[1]
    has_det_idx = n_cols >= 8

    results = []
    for row in tracks:
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        track_id = int(row[4])
        score    = float(row[5])
        cls      = int(row[6])

        if has_det_idx:
            det_idx = int(row[7])
            if 0 <= det_idx < len(tracker_input):
                # Use the tighter original detector box
                x1, y1, x2, y2 = tracker_input[det_idx, :4]
                score = float(tracker_input[det_idx, 4])
                cls   = int(tracker_input[det_idx, 5])

        results.append((track_id, cls, np.array([x1, y1, x2, y2], dtype=np.float32), score))

    return results


def clamp_box(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """
    Convert xyxy → xywh, clamping the origin to ≥ 0 and adjusting the
    width/height consistently so the far edge stays correct.
    """
    x = max(0.0, x1)
    y = max(0.0, y1)
    w = max(0.0, x2 - x)   # use clamped x, not raw x1
    h = max(0.0, y2 - y)
    return x, y, w, h


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    source_path   = Path(SOURCE)
    sequence_name = source_path.name

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_txt_path     = unique_path(OUTPUT_DIR / f"INFERENCE_MOT11_CVAT_{sequence_name}.txt")
    labels_txt_path = unique_path(OUTPUT_DIR / f"INFERENCE_MOT11_CVAT_{sequence_name}_labels.txt")
    zip_path        = unique_path(OUTPUT_DIR / f"INFERENCE_MOT11_CVAT_{sequence_name}.zip")

    # ------------------------------------------------------------------
    # Load model + build class map
    # ------------------------------------------------------------------
    model = YOLO(MODEL_PATH)

    if isinstance(model.names, dict):
        class_names = [model.names[i] for i in sorted(model.names.keys())]
    else:
        class_names = list(model.names)

    # MOT class IDs are 1-indexed
    class_id_map: dict[int, int] = {i: i + 1 for i in range(len(class_names))}

    # ------------------------------------------------------------------
    # Collect images + build tracker
    # ------------------------------------------------------------------
    image_paths = collect_images(source_path)
    tracker     = build_tracker(FRAME_RATE)

    print(f"Source      : {source_path}")
    print(f"Images found: {len(image_paths)}")
    print(f"Classes     : {class_names}")
    print(f"Device      : {DEVICE}")
    print()

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    mot_lines: list[str] = []
    frames_with_tracks = 0

    for frame_idx, img_path in enumerate(image_paths, start=2):     # i do not understand why it shall start at 2 but it is correct
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[WARN] Could not read: {img_path}")
            continue

        # --- Detect ---------------------------------------------------
        pred = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
            save=False,
            show=False,
            device=DEVICE,
        )[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            if frame_idx % LOG_INTERVAL == 0 or frame_idx == len(image_paths):
                print(f"  [{frame_idx:>5}/{len(image_paths)}] no detections")
            continue

        # BYTETracker.update() expects a Boxes object — it calls .conf on it
        # and uses boolean indexing (results[mask]) internally.
        # We pass pred.boxes directly, and separately build a numpy array
        # for original-box recovery in parse_tracks().
        boxes_cpu     = pred.boxes.cpu()
        tracker_input = detections_to_tracker_input(boxes_cpu)  # for box recovery

        # --- Track ----------------------------------------------------
        tracks = tracker.update(boxes_cpu, frame)

        if tracks is None or len(tracks) == 0:
            if frame_idx % LOG_INTERVAL == 0 or frame_idx == len(image_paths):
                print(f"  [{frame_idx:>5}/{len(image_paths)}] no tracks")
            continue

        # --- Parse + write MOT rows -----------------------------------
        parsed = parse_tracks(tracks, tracker_input)
        frames_with_tracks += 1

        # CVAT expects 0-based sequential frame IDs (0 to N-1)
        frame_id = frame_idx - 1

        for track_id, cls_idx, xyxy, _ in parsed:
            x, y, w, h = clamp_box(*xyxy)
            mot_class_id = class_id_map.get(cls_idx, cls_idx + 1)

            mot_lines.append(
                f"{frame_id},"
                f"{track_id},"
                f"{x:.2f},"
                f"{y:.2f},"
                f"{w:.2f},"
                f"{h:.2f},"
                f"1,"                      # not_ignored
                f"{mot_class_id},"
                f"1.000000"                # visibility
            )

        if frame_idx % LOG_INTERVAL == 0 or frame_idx == len(image_paths):
            print(f"  [{frame_idx:>5}/{len(image_paths)}]  tracks: {len(parsed):>3}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    gt_txt_path.write_text("\n".join(mot_lines), encoding="utf-8")

    labels_txt_path.write_text(
        "\n".join(class_names), encoding="utf-8"
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(gt_txt_path,     arcname="gt/gt.txt")
        zf.write(labels_txt_path, arcname="gt/labels.txt")

    print()
    print("=" * 50)
    print(f"Frames total         : {len(image_paths)}")
    print(f"Frames with tracks   : {frames_with_tracks}")
    print(f"Total MOT rows       : {len(mot_lines)}")
    print(f"GT txt               : {gt_txt_path}")
    print(f"Labels txt           : {labels_txt_path}")
    print(f"CVAT zip             : {zip_path}")


if __name__ == "__main__":
    main()
"""
Run SAM3 semantic segmentation on every image in a directory, extract bounding
boxes from the resulting masks, save them in YOLO format, and produce an output
video with the bounding boxes rendered.

Usage:
    python sam3_batch_detect.py [IMAGE_DIR] [--output OUTPUT_DIR]
                                [--model MODEL_PATH] [--conf CONF]
                                [--query QUERY [QUERY ...]]
                                [--fps FPS] [--threshold THRESHOLD]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
DEFAULT_IMAGE_DIR = (
    "/cluster/home/henrban/aquaculture-perception/data-processing/vision"
    "/SOLAQUA/raw_processed/all_images/2024-08-20_17-39-32"
)
DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "sam3.pt"
)


def masks_to_yolo_boxes(masks: np.ndarray, img_h: int, img_w: int, threshold: float = 0.5):
    """Convert binary/float mask array (N, H, W) to YOLO bounding boxes.

    Returns list of (cx, cy, w, h) tuples, all normalised to [0, 1].
    Masks whose bounding box has zero area are skipped.
    """
    boxes = []
    for m in masks:
        binary = (m > threshold).astype(np.uint8)
        if binary.shape[0] != img_h or binary.shape[1] != img_w:
            binary = cv2.resize(binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(binary)
        if len(xs) == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        nw = bw / img_w
        nh = bh / img_h
        boxes.append((cx, cy, nw, nh))
    return boxes


def draw_boxes(img: np.ndarray, boxes, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    vis = img.copy()
    for cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis


def build_predictor(model_path: str, conf: float) -> SAM3SemanticPredictor:
    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,
        save=False,
        show=False,
        verbose=False,
    )
    return SAM3SemanticPredictor(overrides=overrides)


def process_images(
    image_dir: Path,
    output_dir: Path,
    model_path: str,
    query_texts: list,
    conf: float,
    mask_threshold: float,
    fps: float,
):
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} images. Running SAM3...")
    predictor = build_predictor(model_path, conf)

    # Determine video dimensions from first image
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        print(f"Could not read {image_paths[0]}")
        sys.exit(1)
    img_h, img_w = first.shape[:2]

    video_path = str(output_dir / "detections.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (img_w, img_h))

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] skipping unreadable image: {img_path.name}")
            writer.write(np.zeros((img_h, img_w, 3), dtype=np.uint8))
            continue

        try:
            predictor.set_image(str(img_path))
            results = predictor(text=query_texts)
            r = results[0]
        except Exception as e:
            print(f"  [WARN] inference failed for {img_path.name}: {e}")
            writer.write(img)
            # Write empty label file
            (labels_dir / (img_path.stem + ".txt")).write_text("")
            continue

        h, w = img.shape[:2]

        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()  # (N, H, W)
            boxes = masks_to_yolo_boxes(masks, h, w, threshold=mask_threshold)
        else:
            boxes = []

        # Save YOLO label file (class 0 = fish)
        label_file = labels_dir / (img_path.stem + ".txt")
        lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes]
        label_file.write_text("\n".join(lines))

        # Draw boxes and write video frame
        frame = draw_boxes(img, boxes)
        writer.write(frame)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            print(f"  {idx + 1}/{len(image_paths)}  last: {img_path.name}  boxes: {len(boxes)}")

    writer.release()
    print(f"\nDone.")
    print(f"  Labels: {labels_dir}")
    print(f"  Video:  {video_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 batch fish detector -> YOLO + video")
    parser.add_argument(
        "image_dir",
        nargs="?",
        default=DEFAULT_IMAGE_DIR,
        help="Directory of input images",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <image_dir>/../../sam3_detections)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to sam3.pt")
    parser.add_argument(
        "--query", nargs="+", default=["fish"], help="Text query for SAM3 (default: fish)"
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Mask binarisation threshold"
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS")
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_dir = image_dir.parent / (image_dir.name + "_sam3_detections")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print(f"Image dir : {image_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Model     : {model_path}")
    print(f"Query     : {args.query}")
    print(f"Conf      : {args.conf}  Mask threshold: {args.threshold}  FPS: {args.fps}")

    process_images(
        image_dir=image_dir,
        output_dir=output_dir,
        model_path=str(model_path),
        query_texts=args.query,
        conf=args.conf,
        mask_threshold=args.threshold,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

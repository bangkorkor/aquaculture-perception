import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from typing import Union, Optional, Sequence
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import glob
from tqdm import tqdm



def plot_evaluation_metrics(run_dir: str):
    """
    Plot YOLO evaluation curves and confusion matrices from a training run directory.

    Parameters
    ----------
    run_dir : str
        Path to a YOLO 'runs' evaluation folder.
        Expected files:
            - BoxPR_curve.png
            - BoxP_curve.png
            - BoxR_curve.png
            - BoxF1_curve.png
            - confusion_matrix.png
            - confusion_matrix_normalized.png
    """

    # Files to look for in the run folder
    to_show = [
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ]

    # Load available images
    images = []
    titles = []
    for fname in to_show:
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            try:
                img = mpimg.imread(path)
                images.append(img)
                titles.append(fname)
            except Exception as e:
                print(f"⚠️ Could not read {fname}: {e}")
        else:
            print(f"⚠️ Missing: {fname}")

    if not images:
        print("No evaluation images found in:", run_dir)
        return

    # --- Plot grid ---
    cols = 3
    rows = (len(images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()




def plot_yolo_predictions_samples(
    images_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    weights: Union[str, Path],
    num_samples: int = 6,
    imgsz: int = 640,
    conf_thres: float = 0.25
) -> None:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    model = YOLO(str(weights))
    names = model.names  # may be dict or list

    def yolo_txt_to_xyxy(txt_line, img_w, img_h):
        c, cx, cy, w, h = txt_line.strip().split()
        c = int(float(c))
        cx, cy, w, h = map(float, (cx, cy, w, h))
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        return c, int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def draw_boxes(img, boxes, labels=None, color=(255, 0, 0), thickness=2):
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            if labels is not None:
                txt = labels[i]
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ytxt = max(0, y1 - 4)
                cv2.rectangle(img, (x1, ytxt - th - 4), (x1 + tw + 4, ytxt), color, -1)
                cv2.putText(img, txt, (x1 + 2, ytxt - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return img

    # collect images
    image_paths = sorted([p for p in images_dir.rglob("*")
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
    random.shuffle(image_paths)
    image_paths = image_paths[:num_samples]

    cols = 2
    rows = len(image_paths)
    plt.figure(figsize=(12, 4 * rows))

    for idx, img_path in enumerate(image_paths, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠️ Could not read {img_path}")
            continue
        h, w = img_bgr.shape[:2]

        # ground truth
        gt_boxes, gt_labels = [], []
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c, x1, y1, x2, y2 = yolo_txt_to_xyxy(line, w, h)
                    gt_boxes.append((x1, y1, x2, y2))
                    # names can be dict or list
                    cls_name = names[c] if isinstance(names, (list, tuple)) else names.get(c, str(c))
                    gt_labels.append(cls_name)
        gt_img = draw_boxes(img_bgr.copy(), gt_boxes, gt_labels, color=(0, 200, 0))

        # predictions
        res = model.predict(str(img_path), imgsz=imgsz, conf=conf_thres, verbose=False)[0]
        pred_boxes, pred_labels = [], []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            cls  = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                pred_boxes.append((x1, y1, x2, y2))
                cls_name = names[c] if isinstance(names, (list, tuple)) else names.get(c, str(c))
                pred_labels.append(f"{cls_name} {p:.2f}")
        pred_img = draw_boxes(img_bgr.copy(), pred_boxes, pred_labels, color=(0, 0, 255))

        # plot (BGR→RGB)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        pr_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, 2*(idx-1)+1)
        plt.imshow(gt_rgb); plt.title(f"GT: {img_path.name}"); plt.axis("off")

        plt.subplot(rows, cols, 2*(idx-1)+2)
        plt.imshow(pr_rgb); plt.title(" Actual (green) | Predictions (red)"); plt.axis("off")

    plt.tight_layout()
    plt.show()


def annotate_images_from_folder(
    images_dir: Union[str, Path],
    out_dir: Union[str, Path],
    weights: Union[str, Path],
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: Optional[str] = None,         # e.g., "0" for GPU, "cpu" for CPU, None = auto
    classes: Optional[Sequence[int]] = None,  # e.g., [0,1] to restrict classes
    pattern: str = "*.jpg",               # change to "*.png" or "*.jpg" as needed
) -> None:
    """
    Run YOLO predictions image-by-image and save annotated images with the same filenames.
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    model = YOLO(str(weights))
    names = model.names  # can be list/tuple or dict

    # Collect images
    paths = sorted(images_dir.glob(pattern))
    if not paths:
        print(f"No images found in {images_dir} matching {pattern}")
        return

    # Simple deterministic color palette per class
    def cls_color(c: int) -> tuple:
        # nice distinct-ish palette
        palette = [
            (255,  56,  56), (255, 159,  56), (255, 255,  56),
            ( 56, 255,  56), ( 56, 255, 255), ( 56,  56, 255),
            (255,  56, 255), (180, 130,  70), ( 80, 175,  76),
        ]
        return palette[c % len(palette)]

    def draw_boxes(img, boxes, labels):
        h, w = img.shape[:2]
        thick = max(2, int(round(0.0025 * (h + w) * 0.5)))
        for (x1, y1, x2, y2), txt, c in boxes:
            color = cls_color(c)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            # label bg box
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytxt = max(0, y1 - 4)
            cv2.rectangle(img, (x1, ytxt - th - 4), (x1 + tw + 6, ytxt), color, -1)
            cv2.putText(img, txt, (x1 + 3, ytxt - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        return img

    print(f"[INFO] Annotating {len(paths)} images from {images_dir}")
    for p in tqdm(paths, desc="YOLO annotate"):
        # Read image
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            tqdm.write(f"⚠️ Could not read {p}")
            continue

        # Predict on a single image
        res = model.predict(
            source=str(p),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
            verbose=False
        )[0]

        # Collect detections
        boxes_to_draw = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            cls  = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, score in zip(xyxy, cls, confs):
                # class name lookup (list/tuple or dict)
                cls_name = names[c] if isinstance(names, (list, tuple)) else names.get(c, str(c))
                label = f"{cls_name} {score:.2f}"
                boxes_to_draw.append(((x1, y1, x2, y2), label, c))

        annotated = img_bgr.copy()
        if boxes_to_draw:
            annotated = draw_boxes(annotated, boxes_to_draw, labels=True)

        # Save with same filename into out_dir
        out_path = out_dir / p.name
        cv2.imwrite(str(out_path), annotated)

    print(f"[DONE] Wrote annotated images to: {out_dir.resolve()}")
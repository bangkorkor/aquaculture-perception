from ultralytics import FastSAM
import numpy as np
import cv2
from pathlib import Path


def overlay_masks(
    source: str,
    model_path: str = "../models/FastSAM-x.pt",
    imgsz: int = 1024,
    conf: float = 0.4,
    iou: float = 0.3,
    device: str = "cpu",
    retina_masks: bool = False,
    alpha: float = 0.5,
    per_instance_colors: bool = True,
    fixed_color_bgr=(0, 255, 0),   # used when per_instance_colors=False
    seed: int = 0,
    threshold: float = 0.5,
    show_count: bool = False,
    count_prefix: str = "Detections",
    out_path: str = "masks_overlay.png",
):
    model = FastSAM(model_path)
    results = model(
        source,
        device=device,
        retina_masks=retina_masks,
        save=False,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )

    r = results[0]
    base = r.orig_img.copy()   # BGR
    H0, W0 = base.shape[:2]

    if r.masks is None:
        raise ValueError("No masks found in results.")

    masks = (r.masks.data > threshold).cpu().numpy().astype(np.uint8)  # (N,H,W)
    n = len(masks)

    overlay = base.copy()

    if per_instance_colors:
        rng = np.random.default_rng(seed)
        colors = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)  # BGR per mask
    else:
        colors = np.tile(np.array(fixed_color_bgr, dtype=np.uint8), (n, 1))

    for i, m in enumerate(masks):
        # Ensure mask matches original image size
        if m.shape[0] != H0 or m.shape[1] != W0:
            m0 = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m0 = m.astype(bool)

        color = colors[i].astype(np.float32)
        region = overlay[m0].astype(np.float32)
        overlay[m0] = (region * (1 - alpha) + color * alpha).astype(np.uint8)

    if show_count:
        text = f"{count_prefix}: {n}"
        cv2.putText(overlay, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), overlay)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {out_path}")
    return str(out_path)


def main():
    # --- config ---
    source = "../../data-processing/vision/FishVideo/segment_test2.png"
    model_path = "../models/FastSAM-s.pt"

    out_multicolor = "../runs/FastSAMs/FastSAMs_fish2_multicolor.png"
    out_singlecolor = "../runs/FastSAMs/FastSAMs_fish2_singlecolor.png"

    # Toggle these (same behaviour as the SAM3 script)
    SHOW_COUNT = True               # <--- set False to hide "Detections: N"
    FIXED_COLOR_BGR = (0, 0, 255)   # <--- used for single-color output (red)

    # 1) Different color per mask
    overlay_masks(
        source,
        model_path=model_path,
        imgsz=1024,
        conf=0.3,
        iou=0.6,
        device="cpu",
        retina_masks=False,
        alpha=0.5,
        per_instance_colors=True,
        seed=0,
        show_count=SHOW_COUNT,
        out_path=out_multicolor,
    )

    # 2) Same color for all masks
    overlay_masks(
        source,
        model_path=model_path,
        imgsz=1024,
        conf=0.3,
        iou=0.3,
        device="cpu",
        retina_masks=False,
        alpha=0.5,
        per_instance_colors=False,
        fixed_color_bgr=FIXED_COLOR_BGR,
        show_count=SHOW_COUNT,
        out_path=out_singlecolor,
    )

    print("Saved:")
    print(" -", out_multicolor)
    print(" -", out_singlecolor)


if __name__ == "__main__":
    main()

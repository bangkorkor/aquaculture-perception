from ultralytics.models.sam import SAM3SemanticPredictor
import numpy as np
import cv2
from pathlib import Path

def save_masks_overlay_from_results(
    r,
    out_path: str,
    alpha: float = 0.5,
    per_instance_colors: bool = True,
    fixed_color_bgr=(0, 255, 0),
    seed: int = 0,
    threshold: float = 0.5,
    show_count: bool = False,
    count_prefix: str = "Detections",
):
    """Save original image with mask overlay only (no boxes/labels).
    Optionally writes number of detections (masks) in top-left.
    """
    if r.masks is None:
        raise ValueError("No masks found in results.")

    base = r.orig_img.copy()  # BGR
    H0, W0 = base.shape[:2]

    masks = (r.masks.data > threshold).cpu().numpy().astype(np.uint8)  # (N,H,W)
    n = len(masks)

    overlay = base.copy()

    if per_instance_colors:
        rng = np.random.default_rng(seed)
        colors = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
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
    image_path = "../../data-processing/vision/FishVideo/segment_test1.png"
    model_path = "../models/sam3.pt"
    query_texts = ["Fish"]

    out_multicolor = "../runs/SAM3/SAM3_fish1_multicolor.png"
    out_singlecolor = "../runs/SAM3/SAM3_fish1_singlecolor.png"

    # Toggle these
    SHOW_COUNT = True               # <--- set False to hide "Detections: N"
    PER_INSTANCE_COLORS = True      # <--- True = unique color per mask
    FIXED_COLOR_BGR = (0, 0, 255)   # <--- used when PER_INSTANCE_COLORS=False (red)

    # --- predictor ---
    overrides = dict(
        conf=0.30,
        task="segment",
        mode="predict",
        model=model_path,
        half=True,
        save=False,
        show=False,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    predictor.set_image(image_path)

    results = predictor(text=query_texts)
    r = results[0]

    # 1) Multicolor instance masks
    save_masks_overlay_from_results(
        r,
        out_path=out_multicolor,
        per_instance_colors=True,
        seed=0,
        alpha=0.5,
        show_count=SHOW_COUNT,
    )

    # 2) Single color for all masks
    save_masks_overlay_from_results(
        r,
        out_path=out_singlecolor,
        per_instance_colors=False,
        fixed_color_bgr=FIXED_COLOR_BGR,
        alpha=0.5,
        show_count=SHOW_COUNT,
    )

    print("Saved:")
    print(" -", out_multicolor)
    print(" -", out_singlecolor)


if __name__ == "__main__":
    main()

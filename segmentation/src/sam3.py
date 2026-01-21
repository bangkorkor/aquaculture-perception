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
):
    """Save original image with mask overlay only (no boxes/labels)."""
    if r.masks is None:
        raise ValueError("No masks found in results.")

    base = r.orig_img.copy()  # BGR
    H0, W0 = base.shape[:2]

    masks = r.masks.data  # torch tensor (N,H,W)
    masks = (masks > threshold).cpu().numpy().astype(np.uint8)

    overlay = base.copy()

    if per_instance_colors:
        rng = np.random.default_rng(seed)
        colors = rng.integers(0, 256, size=(len(masks), 3), dtype=np.uint8)
    else:
        colors = np.tile(np.array(fixed_color_bgr, dtype=np.uint8), (len(masks), 1))

    for i, m in enumerate(masks):
        # Ensure mask matches original image size
        if m.shape[0] != H0 or m.shape[1] != W0:
            m0 = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m0 = m.astype(bool)

        color = colors[i].astype(np.float32)
        region = overlay[m0].astype(np.float32)
        overlay[m0] = (region * (1 - alpha) + color * alpha).astype(np.uint8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), overlay)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for: {out_path}")
    return str(out_path)

# --- Your SAM3 code, with save=False and manual overlay saving ---
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="../models/sam3.pt",
    half=True,
    save=False,   # IMPORTANT: don't let Ultralytics auto-save annotated outputs
    show=False,
    verbose=False,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

predictor.set_image("../../data-processing/vision/segment_test.png")

results = predictor(text=["Fish"])
r = results[0]

# 1) Multicolor instance masks
save_masks_overlay_from_results(
    r,
    out_path="../runs/SAM3_fish_multicolor.png",
    per_instance_colors=True,
    seed=0,
    alpha=0.5,
)

# 2) Single color for all masks (red)
save_masks_overlay_from_results(
    r,
    out_path="../runs/SAM3_fish_singlecolor.png",
    per_instance_colors=False,
    fixed_color_bgr=(0, 0, 255),
    alpha=0.5,
)

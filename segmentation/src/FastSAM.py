from ultralytics import FastSAM
import numpy as np
import cv2

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
        iou=iou
    )

    r = results[0]
    base = r.orig_img.copy()   # BGR
    H0, W0 = base.shape[:2]

    masks = (r.masks.data > 0.5).cpu().numpy().astype(np.uint8)  # (N,H,W)

    overlay = base.copy()

    if per_instance_colors:
        rng = np.random.default_rng(seed)
        colors = rng.integers(0, 256, size=(len(masks), 3), dtype=np.uint8)  # BGR per mask
    else:
        colors = np.tile(np.array(fixed_color_bgr, dtype=np.uint8), (len(masks), 1))

    for i, m in enumerate(masks):
        m0 = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)

        color = colors[i].astype(np.float32)
        region = overlay[m0].astype(np.float32)
        overlay[m0] = (region * (1 - alpha) + color * alpha).astype(np.uint8)

    cv2.imwrite(out_path, overlay)
    return out_path

# --- Example usage ---
source = "../../data-processing/vision/segment_test.png"

# 1) Different color per mask
overlay_masks(source, per_instance_colors=True, seed=0, out_path="../runs/FastSAMx_multicolor.png")

# 2) Same color for all masks (red in BGR)
overlay_masks(source, per_instance_colors=False, fixed_color_bgr=(0, 0, 255), out_path="../runs/FastSAMx_singlecolor.png")

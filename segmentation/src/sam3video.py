from ultralytics.models.sam import SAM3VideoSemanticPredictor
import numpy as np
import cv2
from pathlib import Path

def overlay_from_results(
    r,
    alpha: float = 0.5,
    fixed_color_bgr=(0, 255, 0),
    seed: int = 0,
    threshold: float = 0.5,
    use_track_ids_for_colors: bool = True,
):
    """
    Return a BGR image = original frame + mask overlay only (no boxes/labels).
    Tries to keep colors consistent across frames if track IDs exist.
    """
    if r.masks is None:
        return r.orig_img  # no masks this frame

    base = r.orig_img.copy()  # BGR
    H0, W0 = base.shape[:2]

    masks = r.masks.data  # torch tensor (N,h,w)
    masks = (masks > threshold).cpu().numpy().astype(np.uint8)
    n = len(masks)
    if n == 0:
        return base

    # Get track IDs if available (helps stable colors frame-to-frame)
    ids = None
    if use_track_ids_for_colors and getattr(r, "boxes", None) is not None:
        bid = getattr(r.boxes, "id", None)
        if bid is not None:
            ids = bid.detach().cpu().numpy().astype(int).tolist()  # length N (often)

    overlay = base.copy()

    def color_for(k: int):
        # deterministic color from integer key
        rng = np.random.default_rng(seed + int(k) * 10007)
        return rng.integers(0, 256, size=(3,), dtype=np.uint8)

    for i, m in enumerate(masks):
        # ensure mask matches original frame size
        if m.shape[0] != H0 or m.shape[1] != W0:
            m0 = cv2.resize(m, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m0 = m.astype(bool)

        if ids is not None and i < len(ids):
            c = color_for(ids[i])
        else:
            c = color_for(i)  # consistent per-index if no ids
        c = c.astype(np.float32)

        region = overlay[m0].astype(np.float32)
        overlay[m0] = (region * (1 - alpha) + c * alpha).astype(np.uint8)

    return overlay


from ultralytics.models.sam import SAM3VideoSemanticPredictor
import numpy as np
import cv2
from pathlib import Path
import time  # <-- add

# ... keep overlay_from_results exactly as you have it ...

def sam3_text_video_overlay(
    video_path: str,
    out_video_path: str,
    text_prompts: list[str],
    model_path: str = "sam3.pt",
    imgsz: int = 640,
    conf: float = 0.25,
    half: bool = True,
    alpha: float = 0.5,
    threshold: float = 0.5,
    seed: int = 0,
    print_every: int = 25,  # <-- add (how often to print)
):
    video_path = str(video_path)
    out_video_path = Path(out_video_path)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # <-- add
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_video_path}")

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        imgsz=imgsz,
        model=model_path,
        half=half,
        save=False,
        show=False,
        verbose=False,
        stream_buffer=False,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)

    results = predictor(source=video_path, text=text_prompts, stream=True)

    # --- progress counters (minimal) ---
    frames_done = 0
    frames_with_masks = 0
    t0 = time.time()

    try:
        for r in results:
            frames_done += 1

            # count frames that actually got masks
            if r.masks is not None and getattr(r.masks, "data", None) is not None:
                if int(r.masks.data.shape[0]) > 0:
                    frames_with_masks += 1

            frame_overlay = overlay_from_results(
                r,
                alpha=alpha,
                seed=seed,
                threshold=threshold,
                use_track_ids_for_colors=True,
            )

            if frame_overlay.shape[0] != H or frame_overlay.shape[1] != W:
                frame_overlay = cv2.resize(frame_overlay, (W, H), interpolation=cv2.INTER_LINEAR)

            writer.write(frame_overlay)

            # print progress every N frames (and at the end if total known)
            if (frames_done % print_every) == 0 or (total_frames and frames_done == total_frames):
                dt = time.time() - t0
                proc_fps = frames_done / max(dt, 1e-6)
                if total_frames:
                    pct = 100.0 * frames_done / total_frames
                    print(
                        f"[SAM3] {frames_done}/{total_frames} ({pct:.1f}%) | "
                        f"frames_with_masks={frames_with_masks} | proc_fps={proc_fps:.2f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SAM3] {frames_done} frames | frames_with_masks={frames_with_masks} | "
                        f"proc_fps={proc_fps:.2f}",
                        flush=True,
                    )

    finally:
        writer.release()

    print(f"[SAM3] DONE | frames={frames_done} | frames_with_masks={frames_with_masks}", flush=True)
    return str(out_video_path)


if __name__ == "__main__":
    out = sam3_text_video_overlay(
        video_path="../../data-processing/vision/FishVideo/FishVideo_12sec.mp4",
        out_video_path="../runs/sam3_fish_overlay.mp4",
        text_prompts=["Fish"],     # <-- your concept(s)
        model_path="../models/sam3.pt",
        imgsz=640,
        conf=0.25,
        half=True,
        alpha=0.5,
        threshold=0.5,
        seed=0,
    )
    print("Saved:", out)

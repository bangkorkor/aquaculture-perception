#!/usr/bin/env python3
"""
Render all inference .txt results onto their source images and save MP4 videos.

Scans outputs/inference_annotation_MOT11/ for files matching:
  INFERENCE_MOT11_{seq}_{model}+{tracker}.txt          -> vision
  INFERENCE_MOT11_SONAR_{seq}_{model}+{tracker}.txt    -> sonar

Saves one MP4 per run to:
  outputs/inference_mp4/{modality}_{seq}_{model}+{tracker}.mp4

Run from the tracking/ directory:
    python render_inference_mp4.py
    python render_inference_mp4.py --show_frame_id
    python render_inference_mp4.py --show_utc_datetime
"""

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── Paths (relative to this script) ───────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent
REPO_ROOT      = _HERE.parent
MOT_OUTPUT_DIR = _HERE / "outputs" / "inference_annotation_MOT11"
VIDEO_OUT_DIR  = _HERE / "outputs" / "inference_mp4"
VISION_MOT     = REPO_ROOT / "data-processing" / "vision" / "MOT"
SONAR_MOT      = REPO_ROOT / "data-processing" / "sonar"  / "MOT"

FPS        = "auto"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_SEQ_RE    = r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
_VISION_PAT = re.compile(rf"^INFERENCE_MOT11_{_SEQ_RE}_(.+\+.+)\.txt$")
_SONAR_PAT  = re.compile(rf"^INFERENCE_MOT11_SONAR_{_SEQ_RE}_(.+\+.+)\.txt$")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show_frame_id",    action="store_true",
                   help="Overlay nanosecond timestamp in bottom-right corner.")
    p.add_argument("--show_utc_datetime", action="store_true",
                   help="Overlay UTC datetime converted from nanosecond timestamp.")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_mot_txt(txt_path):
    """Return {frame_id_str: [{'track_id', 'x', 'y', 'w', 'h'}, ...]}."""
    dets = {}
    with open(txt_path, newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) < 6:
                continue
            frame_key = row[0].strip()
            dets.setdefault(frame_key, []).append({
                "track_id": int(float(row[1])),
                "x": float(row[2]), "y": float(row[3]),
                "w": float(row[4]), "h": float(row[5]),
            })
    return dets


def list_images(img_dir):
    imgs = [p for p in Path(img_dir).iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS]
    imgs.sort(key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
    return imgs


def auto_fps(image_paths):
    if len(image_paths) < 2:
        return 10.0
    try:
        ts = [int(p.stem) for p in image_paths]
    except ValueError:
        return 10.0
    fps = (len(ts) - 1) * 1e9 / (ts[-1] - ts[0])
    return fps if 0 < fps <= 240 else 10.0


def get_color(track_id):
    rng = np.random.default_rng(seed=int(track_id))
    c = rng.integers(50, 256, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])


def draw_label(img, text, x, y, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x, y - th - bl - 4), (x + tw + 6, y + 4), color, -1)
    cv2.putText(img, text, (x + 3, y - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_bottom_right(img, lines):
    if not lines:
        return
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    sizes   = [cv2.getTextSize(ln, font, scale, thick) for ln in lines]
    box_w   = max(s[0][0] for s in sizes) + 12
    box_h   = sum(s[0][1] + s[1] for s in sizes) + 8 * (len(lines) - 1) + 12
    x1, y2  = img.shape[1] - box_w - 10, img.shape[0] - 10
    cv2.rectangle(img, (x1, y2 - box_h), (x1 + box_w, y2), (0, 0, 0), -1)
    cy = y2 - box_h + 8
    for ln, ((tw, th), bl) in zip(lines, sizes):
        cv2.putText(img, ln, (img.shape[1] - tw - 16, cy + th),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        cy += th + bl + 8


def ns_to_utc(ns_str):
    ts = int(ns_str)
    sec, ns = ts // 1_000_000_000, ts % 1_000_000_000
    return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") \
           + f".{ns:09d} UTC"


# ── Core render ───────────────────────────────────────────────────────────────

def render_run(txt_path, img_dir, out_path, show_frame_id, show_utc):
    """Render one inference .txt onto images and write MP4. Returns True on success."""
    imgs = list_images(img_dir)
    if not imgs:
        print(f"  [SKIP] no images found in {img_dir}")
        return False

    dets  = load_mot_txt(txt_path)
    first = cv2.imread(str(imgs[0]))
    if first is None:
        print(f"  [SKIP] cannot read {imgs[0]}")
        return False

    h, w = first.shape[:2]
    fps  = auto_fps(imgs) if FPS == "auto" else float(FPS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n = len(imgs)
    for i, img_path in enumerate(imgs, 1):
        frame_key = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] cannot read {img_path.name}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        for det in dets.get(frame_key, []):
            tid = det["track_id"]
            x1 = int(round(det["x"]));            y1 = int(round(det["y"]))
            x2 = int(round(det["x"] + det["w"])); y2 = int(round(det["y"] + det["h"]))
            color = get_color(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            draw_label(img, f"ID {tid}", x1, max(y1 - 5, 20), color)

        overlay = []
        if show_frame_id:
            overlay.append(frame_key)
        if show_utc:
            try:
                overlay.append(ns_to_utc(frame_key))
            except ValueError:
                overlay.append("Invalid timestamp")
        if overlay:
            draw_bottom_right(img, overlay)

        writer.write(img)
        if i % 200 == 0 or i == n:
            print(f"    {i}/{n} frames")

    writer.release()
    return True


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_runs():
    """Yield (modality, seq, tag, txt_path, img_dir) for every valid inference .txt."""
    for txt in sorted(MOT_OUTPUT_DIR.glob("INFERENCE_MOT11_*.txt")):
        name = txt.name

        m = _SONAR_PAT.match(name)
        if m:
            seq, tag = m.group(1), m.group(2)
            yield "sonar", seq, tag, txt, SONAR_MOT / seq / "frames"
            continue

        m = _VISION_PAT.match(name)
        if m:
            seq, tag = m.group(1), m.group(2)
            yield "vision", seq, tag, txt, VISION_MOT / seq / "frames" / seq


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    runs = list(discover_runs())
    if not runs:
        print(f"No matching inference .txt files found in:\n  {MOT_OUTPUT_DIR}")
        return

    print(f"Found {len(runs)} inference run(s)")
    print(f"Output folder: {VIDEO_OUT_DIR}\n")

    ok = fail = 0
    for modality, seq, tag, txt, img_dir in runs:
        out_path = VIDEO_OUT_DIR / f"{modality}_{seq}_{tag}.mp4"
        print(f"[{modality}] {seq}  {tag}")
        print(f"  txt : {txt.name}")
        print(f"  imgs: {img_dir}")
        print(f"  out : {out_path.name}")

        if not img_dir.is_dir():
            print(f"  [SKIP] image dir not found")
            fail += 1
            print()
            continue

        success = render_run(txt, img_dir, out_path, args.show_frame_id, args.show_utc_datetime)
        if success:
            size_mb = out_path.stat().st_size / 1_048_576
            print(f"  saved ({size_mb:.1f} MB)")
            ok += 1
        else:
            fail += 1
        print()

    print(f"Done — {ok} saved, {fail} skipped.")


if __name__ == "__main__":
    main()

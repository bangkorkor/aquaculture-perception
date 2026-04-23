#!/usr/bin/env python3
"""
Plot time differences between consecutive frame files whose filenames are timestamps.

Assumes filenames look like:
    1724157091855545200.jpg

These are treated as integer timestamps, typically nanoseconds.
The script plots consecutive frame deltas in milliseconds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# HARD CODE PATHS HERE
# =========================
IMAGE_DIR = "/cluster/home/henrban/aquaculture-perception/data-processing/vision/SOLAQUA/raw_processed/all_images/2024-08-20_14-31-29"
OUTPUT_PLOT = "/cluster/home/henrban/aquaculture-perception/tracking/outputs/fps/frame_time_deltas.png"
OUTPUT_HIST = "/cluster/home/henrban/aquaculture-perception/tracking/outputs/varying_fps_plots/frame_time_deltas_hist.png"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def numeric_sort_key(path_obj):
    try:
        return int(path_obj.stem)
    except ValueError:
        return path_obj.stem


def load_timestamps_from_filenames(image_dir):
    image_dir = Path(image_dir)

    image_paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    image_paths.sort(key=numeric_sort_key)

    timestamps = []
    bad_files = []

    for p in image_paths:
        try:
            timestamps.append(int(p.stem))
        except ValueError:
            bad_files.append(p.name)

    return image_paths, np.array(timestamps, dtype=np.int64), bad_files


def main():
    image_paths, timestamps, bad_files = load_timestamps_from_filenames(IMAGE_DIR)

    if bad_files:
        print("Skipped non-numeric filenames:")
        for name in bad_files[:20]:
            print(f"  {name}")
        if len(bad_files) > 20:
            print(f"  ... and {len(bad_files) - 20} more")

    if len(timestamps) < 2:
        raise RuntimeError("Need at least 2 timestamp-named images to compute differences.")

    # consecutive differences
    delta_ns = np.diff(timestamps)
    delta_ms = delta_ns / 1e6
    frame_idx = np.arange(1, len(timestamps))

    # stats
    mean_ms = float(np.mean(delta_ms))
    median_ms = float(np.median(delta_ms))
    min_ms = float(np.min(delta_ms))
    max_ms = float(np.max(delta_ms))
    std_ms = float(np.std(delta_ms))

    # estimated fps from median delta
    est_fps = 1000.0 / median_ms if median_ms > 0 else float("inf")

    print(f"Found {len(image_paths)} images")
    print(f"Computed {len(delta_ms)} frame-to-frame deltas")
    print()
    print("Delta statistics (milliseconds):")
    print(f"  mean   = {mean_ms:.3f} ms")
    print(f"  median = {median_ms:.3f} ms")
    print(f"  min    = {min_ms:.3f} ms")
    print(f"  max    = {max_ms:.3f} ms")
    print(f"  std    = {std_ms:.3f} ms")
    print(f"  estimated FPS from median delta = {est_fps:.3f}")

    # line plot
    plt.figure(figsize=(12, 5))
    plt.plot(frame_idx, delta_ms)
    plt.xlabel("Frame index")
    plt.ylabel("Delta to previous frame [ms]")
    plt.title("Time difference between consecutive frames")
    plt.grid(True)
    plt.tight_layout()
    Path(OUTPUT_PLOT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(delta_ms, bins=50)
    plt.xlabel("Delta between frames [ms]")
    plt.ylabel("Count")
    plt.title("Histogram of frame-to-frame time differences")
    plt.grid(True)
    plt.tight_layout()
    Path(OUTPUT_HIST).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_HIST, dpi=150)
    plt.close()

    print()
    print(f"Saved line plot to: {OUTPUT_PLOT}")
    print(f"Saved histogram to: {OUTPUT_HIST}")


if __name__ == "__main__":
    main()
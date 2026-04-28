#!/usr/bin/env python3

# visualize MOT_format files on corresponding images
# input is image files from video sequence (vision/sonar) and the annotated gt file
# output is a mp4 showing mp4
# it also has the option to show the timestamp/frame-id in the bottom right corner.

"""
Visualize MOT-format annotations on corresponding image frames and export to MP4.

Hardcoded paths:
- IMAGE_DIR
- GT_FILE
- OUTPUT_MP4

Terminal options:
- --show_frame_id
- --show_utc_datetime

Examples:
    python visualize_MOTfile_on_images.py
    python visualize_MOTfile_on_images.py --show_frame_id
    python visualize_MOTfile_on_images.py --show_frame_id --show_utc_datetime
"""

import argparse
import csv
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


# =========================
# HARD CODE PATHS HERE
# =========================
IMAGE_DIR = "/cluster/home/henrban/aquaculture-perception/data-processing/vision/MOT/2024-08-20_14-31-29/frames/2024-08-20_14-31-29"
GT_FILE = "/cluster/home/henrban/aquaculture-perception/data-processing/vision/MOT/2024-08-20_14-31-29/gt/gt.txt"
# GT_FILE = "/cluster/home/henrban/aquaculture-perception/tracking/outputs/inference_annotation_MOT11/INFERENCE_MOT11_2024-08-20_17-02-00.txt"
OUTPUT_MP4 = "/cluster/home/henrban/aquaculture-perception/tracking/outputs/labeled_mp4_demos/vision_GT_demo_2024-08-20_14-31-29.mp4"   # change inference and gt here!! 

# Optional settings
FPS = "auto"   # use "auto" or a number like 10 or 20
SHOW_TRACK_ID = True
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show_frame_id",
        action="store_true",
        help="Show timestamp/frame-id in the bottom-right corner.",
    )
    parser.add_argument(
        "--show_utc_datetime",
        action="store_true",
        help="Show UTC date/time converted from the nanosecond timestamp.",
    )
    return parser.parse_args()


def load_mot_gt(gt_file):
    dets_by_frame = {}

    with open(gt_file, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 6:
                continue

            frame_id = row[0].strip()
            track_id = int(float(row[1]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])

            dets_by_frame.setdefault(frame_id, []).append(
                {
                    "track_id": track_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

    return dets_by_frame


def numeric_sort_key(path_obj):
    stem = path_obj.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def list_images(image_dir):
    image_dir = Path(image_dir)
    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    images.sort(key=numeric_sort_key)
    return images


def infer_average_fps_from_timestamps(image_paths):
    """
    Compute average FPS from the first and last timestamp:

        fps = (N - 1) / total_time_seconds

    Assumes filename stems are nanosecond timestamps.
    """
    if len(image_paths) < 2:
        return 10.0

    try:
        timestamps = [int(p.stem) for p in image_paths]
    except ValueError:
        return 10.0

    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    if last_ts <= first_ts:
        return 10.0

    fps = (len(timestamps) - 1) * 1e9 / (last_ts - first_ts)

    if fps <= 0 or fps > 240:
        return 10.0

    return fps


def ns_timestamp_to_utc_string(timestamp_ns):
    """
    Convert a nanosecond Unix timestamp to a UTC datetime string.

    Example output:
        2024-08-20 12:31:29.123456789 UTC
    """
    timestamp_ns = int(timestamp_ns)
    seconds = timestamp_ns // 1_000_000_000
    nanoseconds = timestamp_ns % 1_000_000_000

    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f".{nanoseconds:09d} UTC"


def get_color(track_id):
    rng = np.random.default_rng(seed=track_id)
    color = rng.integers(50, 256, size=3).tolist()
    return int(color[0]), int(color[1]), int(color[2])


def draw_text_with_bg(img, text, org, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_bottom_right_lines(img, lines):
    """
    Draw one or more lines of text in the bottom-right corner.
    """
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    margin = 10
    line_gap = 8

    text_sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
    text_widths = [size[0][0] for size in text_sizes]
    text_heights = [size[0][1] for size in text_sizes]
    baselines = [size[1] for size in text_sizes]

    box_width = max(text_widths) + 12
    total_text_height = sum(text_heights) + sum(baselines) + line_gap * (len(lines) - 1)
    box_height = total_text_height + 12

    x1 = img.shape[1] - box_width - margin
    y2 = img.shape[0] - margin
    y1 = y2 - box_height

    cv2.rectangle(img, (x1, y1), (x1 + box_width, y2), (0, 0, 0), -1)

    current_y = y1 + 8
    for line, ((tw, th), baseline) in zip(lines, text_sizes):
        text_x = img.shape[1] - tw - margin - 6
        text_y = current_y + th
        cv2.putText(
            img,
            line,
            (text_x, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        current_y = text_y + baseline + line_gap

def get_unique_output_path(path_str):
    """
    Return a unique file path by appending/incrementing a number at the end
    if the file already exists.

    Examples:
        demo.mp4     -> demo.mp4        (if it does not exist)
        demo.mp4     -> demo_1.mp4      (if demo.mp4 exists)
        demo_1.mp4   -> demo_2.mp4      (if demo_1.mp4 exists)
    """
    path = Path(path_str)

    if not path.exists():
        return path

    parent = path.parent
    suffix = path.suffix
    stem = path.stem

    match = re.match(r"^(.*?)(?:_(\d+))?$", stem)
    if match:
        base_name = match.group(1)
        start_num = int(match.group(2)) if match.group(2) else 0
    else:
        base_name = stem
        start_num = 0

    counter = start_num + 1
    while True:
        candidate = parent / f"{base_name}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1



def main():
    args = parse_args()

    image_paths = list_images(IMAGE_DIR)
    if not image_paths:
        raise RuntimeError(f"No images found in: {IMAGE_DIR}")

    dets_by_frame = load_mot_gt(GT_FILE)

    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    height, width = first_img.shape[:2]

    if FPS == "auto":
        fps = infer_average_fps_from_timestamps(image_paths)
    else:
        fps = float(FPS)

    output_path = get_unique_output_path(OUTPUT_MP4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print(f"Found {len(image_paths)} images")
    print(f"Using FPS: {fps:.3f}")
    print(f"Writing video to: {OUTPUT_MP4}")

    if len(image_paths) >= 2:
        first_ts = int(image_paths[0].stem)
        last_ts = int(image_paths[-1].stem)
        print(f"First timestamp UTC: {ns_timestamp_to_utc_string(first_ts)}")
        print(f"Last  timestamp UTC: {ns_timestamp_to_utc_string(last_ts)}")

    for idx, img_path in enumerate(image_paths, start=1):
        frame_key = img_path.stem
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        detections = dets_by_frame.get(frame_key, [])

        for det in detections:
            track_id = det["track_id"]
            x1 = int(round(det["x"]))
            y1 = int(round(det["y"]))
            x2 = int(round(det["x"] + det["w"]))
            y2 = int(round(det["y"] + det["h"]))

            color = get_color(track_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if SHOW_TRACK_ID:
                label = f"ID {track_id}"
                draw_text_with_bg(img, label, (x1, max(y1 - 5, 20)), color)

        bottom_right_lines = []

        if args.show_frame_id:
            bottom_right_lines.append(frame_key)

        if args.show_utc_datetime:
            try:
                bottom_right_lines.append(ns_timestamp_to_utc_string(frame_key))
            except ValueError:
                bottom_right_lines.append("Invalid timestamp")

        if bottom_right_lines:
            draw_bottom_right_lines(img, bottom_right_lines)

        writer.write(img)

        if idx % 100 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)}")

    writer.release()
    print("Done.")


if __name__ == "__main__":
    main()
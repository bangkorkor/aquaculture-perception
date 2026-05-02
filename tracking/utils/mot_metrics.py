"""
tracking/utils/mot_metrics.py
=============================
Shared utilities for ByteTrack inference and MOT evaluation.

Key functions
-------------
run_inference(model_path, img_dir, ...)  -> DataFrame   (same columns as GT)
load_gt(gt_path, img_dir, class_id)     -> DataFrame
compute_metrics(gt_df, pred_df, ...)    -> (accumulator, summary DataFrame)
plot_metrics_bar / plot_error_breakdown / plot_track_comparison / ...
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import motmetrics as mm
import matplotlib.pyplot as plt
from pathlib import Path

import ultralytics
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
GT_COLS = [
    "frame_id", "track_id", "x", "y", "w", "h",
    "not_ignored", "class_id", "visibility",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Use the bytetrack config bundled with the local ultralytics install
BYTETRACK_CFG = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"


# ── Image helpers ─────────────────────────────────────────────────────────────

def collect_images(img_dir: Path | str) -> list[Path]:
    """Return image files sorted numerically by stem (nanosecond timestamps)."""
    paths = [
        p for p in Path(img_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    paths.sort(key=lambda p: int(p.stem))
    return paths


def frame_map(img_dir: Path | str) -> dict[int, int]:
    """Return {timestamp_ns: frame_norm} mapping (1-indexed sequential).
    Image filename stem == nanosecond timestamp == frame_id in gt/gt.txt.
    """
    return {int(f.stem): i + 1 for i, f in enumerate(collect_images(img_dir))}


def img_dir_for(seq_path: Path, modality: str) -> Path:
    """Return the frames directory for a sequence given modality."""
    if modality == "sonar":
        return seq_path / "frames"
    else:  # vision
        return seq_path / "frames" / seq_path.name


# ── Tracker helpers (adapted from tracking-inference_CVAT.py) ─────────────────

def build_tracker(frame_rate: int = 30) -> BYTETracker:
    cfg = IterableSimpleNamespace(**YAML.load(BYTETRACK_CFG))
    return BYTETracker(args=cfg, frame_rate=frame_rate)


def _detections_to_input(boxes) -> np.ndarray:
    return np.column_stack([
        boxes.xyxy.numpy(),
        boxes.conf.numpy(),
        boxes.cls.numpy(),
    ]).astype(np.float32)


def _parse_tracks(tracks: np.ndarray, tracker_input: np.ndarray):
    """Parse ByteTrack rows → list of (track_id, cls_idx, xyxy, conf)."""
    has_det_idx = tracks.shape[1] >= 8
    results = []
    for row in tracks:
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        tid   = int(row[4])
        score = float(row[5])
        cls   = int(row[6])
        if has_det_idx:
            di = int(row[7])
            if 0 <= di < len(tracker_input):
                x1, y1, x2, y2 = tracker_input[di, :4]
                score = float(tracker_input[di, 4])
                cls   = int(tracker_input[di, 5])
        results.append((tid, cls, np.array([x1, y1, x2, y2], dtype=np.float32), score))
    return results


def _clamp_box(x1, y1, x2, y2):
    x = max(0.0, x1); y = max(0.0, y1)
    return x, y, max(0.0, x2 - x), max(0.0, y2 - y)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gt(
    gt_path: Path | str,
    img_dir: Path | str,
    class_id: int | None = None,
) -> pd.DataFrame:
    """Load gt/gt.txt and add frame_norm (1-indexed sequential from image filenames).

    Parameters
    ----------
    gt_path   : path to gt/gt.txt
    img_dir   : folder containing the sequence's .jpg images
    class_id  : if given, keep only rows with this class (1=fish, 2=net)
    """
    ts_fn = frame_map(img_dir)
    df = pd.read_csv(gt_path, header=None, names=GT_COLS)
    df["frame_id"]   = df["frame_id"].astype(np.int64)
    df["frame_norm"] = df["frame_id"].map(ts_fn)
    df = df.dropna(subset=["frame_norm"]).copy()
    df["frame_norm"] = df["frame_norm"].astype(int)
    if class_id is not None:
        df = df[df["class_id"] == class_id].copy()
    return df.reset_index(drop=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model_path: str | Path,
    img_dir: Path | str,
    conf: float = 0.25,
    iou: float  = 0.45,
    device: str = "cuda:0",
    frame_rate: int = 30,
    desc: str = "",
) -> pd.DataFrame:
    """Run detector + ByteTrack on every image in img_dir.

    Returns a DataFrame with GT_COLS + ['frame_norm'].
    frame_id = nanosecond timestamp (image filename stem) — same space as GT.
    """
    model   = YOLO(model_path)
    imgs    = collect_images(img_dir)
    tracker = build_tracker(frame_rate)
    ts_fn   = frame_map(img_dir)

    class_names = (
        [model.names[i] for i in sorted(model.names)]
        if isinstance(model.names, dict)
        else list(model.names)
    )
    cls_map = {i: i + 1 for i in range(len(class_names))}

    rows = []
    for img_path in tqdm(imgs, desc=desc or Path(img_dir).parent.name, leave=True):
        frame_id = int(img_path.stem)
        frame    = cv2.imread(str(img_path))
        if frame is None:
            continue

        pred = model.predict(
            frame, conf=conf, iou=iou, verbose=False, save=False, device=device
        )[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            continue

        boxes_cpu     = pred.boxes.cpu()
        tracker_input = _detections_to_input(boxes_cpu)
        tracks        = tracker.update(boxes_cpu, frame)

        if tracks is None or len(tracks) == 0:
            continue

        fn = ts_fn.get(frame_id)
        for tid, cls_idx, xyxy, _ in _parse_tracks(tracks, tracker_input):
            x, y, w, h = _clamp_box(*xyxy)
            rows.append({
                "frame_id":    frame_id,
                "track_id":    tid,
                "x": x, "y": y, "w": w, "h": h,
                "not_ignored": 1,
                "class_id":    cls_map.get(cls_idx, cls_idx + 1),
                "visibility":  1.0,
                "frame_norm":  fn,
            })

    if not rows:
        return pd.DataFrame(columns=GT_COLS + ["frame_norm"])

    df = pd.DataFrame(rows)
    df["frame_norm"] = pd.to_numeric(df["frame_norm"], errors="coerce")
    df = df.dropna(subset=["frame_norm"]).copy()
    df["frame_norm"] = df["frame_norm"].astype(int)
    return df.reset_index(drop=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _xywh_to_xyxy(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return np.empty((0, 4))
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return np.column_stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]])


METRIC_NAMES = [
    "mota", "motp", "idf1",
    "num_switches", "num_false_positives", "num_misses",
    "num_detections", "num_objects",
    "recall", "precision",
    "mostly_tracked", "mostly_lost", "partially_tracked",
    "num_unique_objects",
]


def compute_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> tuple[mm.MOTAccumulator, pd.DataFrame]:
    """Compute standard MOT metrics (MOTA, MOTP, IDF1, …) via motmetrics.

    Both dataframes must have a 'frame_norm' column (1-indexed int).
    A match is accepted when IoU >= iou_threshold (default 0.5).

    Returns
    -------
    acc     : MOTAccumulator (for per-frame inspection)
    summary : single-row DataFrame with all metrics
    """
    acc = mm.MOTAccumulator(auto_id=False)

    all_frames = sorted(
        set(gt_df["frame_norm"].unique()) | set(pred_df["frame_norm"].unique())
    )

    for fn in all_frames:
        gt_f  = gt_df[gt_df["frame_norm"] == fn]
        pr_f  = pred_df[pred_df["frame_norm"] == fn]

        gt_ids = gt_f["track_id"].values.tolist()
        pr_ids = pr_f["track_id"].values.tolist()

        gt_xyxy = _xywh_to_xyxy(gt_f[["x", "y", "w", "h"]].values)
        pr_xyxy = _xywh_to_xyxy(pr_f[["x", "y", "w", "h"]].values)

        dist = mm.distances.iou_matrix(gt_xyxy, pr_xyxy, max_iou=1 - iou_threshold)
        acc.update(gt_ids, pr_ids, dist, frameid=fn)

    mh      = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRIC_NAMES, name="result")
    return acc, summary


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a human-readable copy with renamed columns and % for rate metrics."""
    rename = {
        "mota": "MOTA", "motp": "MOTP", "idf1": "IDF1",
        "num_switches": "IDSW", "num_false_positives": "FP",
        "num_misses": "FN", "num_detections": "TP",
        "num_objects": "GT dets", "recall": "Recall",
        "precision": "Precision", "mostly_tracked": "MT",
        "mostly_lost": "ML", "partially_tracked": "PT",
        "num_unique_objects": "GT tracks",
    }
    out = summary.rename(columns=rename).copy()
    for col in ["MOTA", "MOTP", "IDF1", "Recall", "Precision"]:
        if col in out.columns:
            out[col] = (out[col] * 100).round(1).astype(str) + "%"
    for col in ["MT", "ML", "PT"]:
        if col in out.columns:
            out[col] = out[col].astype(int)
    return out


def build_summary_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-sequence summaries into one table (sequences as rows)."""
    rows = []
    for seq, s in results.items():
        row = format_summary(s).iloc[0].to_dict()
        row["Sequence"] = seq[-8:]
        rows.append(row)
    cols = ["Sequence", "MOTA", "IDF1", "MOTP", "Recall", "Precision",
            "FP", "FN", "IDSW", "MT", "ML", "GT tracks"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_metrics_bar(
    results: dict[str, pd.DataFrame],
    metrics: tuple = ("MOTA", "IDF1", "MOTP"),
    title: str = "Tracking metrics per sequence",
) -> plt.Figure:
    """Bar chart of selected metrics (as %) across sequences."""
    seq_names = list(results.keys())
    _col = {"MOTA": "mota", "IDF1": "idf1", "MOTP": "motp", "Recall": "recall",
            "Precision": "precision"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        col  = _col.get(m, m.lower())
        vals = [float(results[s][col].iloc[0]) * 100 for s in seq_names]
        colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(seq_names)))
        bars = ax.bar(range(len(seq_names)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(seq_names)))
        ax.set_xticklabels([s[-8:] for s in seq_names], rotation=12, ha="right", fontsize=8)
        ax.set_ylabel(f"{m} (%)")
        ax.set_ylim(0, 105)
        ax.set_title(m, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_error_breakdown(
    results: dict[str, pd.DataFrame],
    title: str = "Error breakdown (normalised by GT detections)",
) -> plt.Figure:
    """Grouped bar chart of FP, FN, IDSW per sequence, each normalised by GT count."""
    seq_names = list(results.keys())
    gt_vals   = [float(results[s]["num_objects"].iloc[0])          for s in seq_names]
    fp_vals   = [float(results[s]["num_false_positives"].iloc[0])  for s in seq_names]
    fn_vals   = [float(results[s]["num_misses"].iloc[0])           for s in seq_names]
    idsw_vals = [float(results[s]["num_switches"].iloc[0])         for s in seq_names]

    fp_n  = [v / g * 100 if g else 0 for v, g in zip(fp_vals,   gt_vals)]
    fn_n  = [v / g * 100 if g else 0 for v, g in zip(fn_vals,   gt_vals)]
    ids_n = [v / g * 100 if g else 0 for v, g in zip(idsw_vals, gt_vals)]

    x = np.arange(len(seq_names)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, fp_n,  w, label="FP",   color="#EF5350", edgecolor="white")
    ax.bar(x,     fn_n,  w, label="FN",   color="#42A5F5", edgecolor="white")
    ax.bar(x + w, ids_n, w, label="IDSW", color="#FFA726", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([s[-8:] for s in seq_names], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("% of GT detections")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_track_comparison(
    gt_dfs: dict[str, pd.DataFrame],
    pred_dfs: dict[str, pd.DataFrame],
) -> plt.Figure:
    """GT vs predicted unique track count per sequence."""
    seq_names   = list(gt_dfs.keys())
    gt_counts   = [gt_dfs[s]["track_id"].nunique()   for s in seq_names]
    pred_counts = [pred_dfs[s]["track_id"].nunique() for s in seq_names]

    x = np.arange(len(seq_names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, gt_counts,   w, label="GT",   color="#2E7D32", edgecolor="white")
    ax.bar(x + w / 2, pred_counts, w, label="Pred", color="#1565C0", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([s[-8:] for s in seq_names], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Unique tracks")
    ax.set_title("GT vs predicted unique tracks", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_det_timeline(
    gt_dfs: dict[str, pd.DataFrame],
    pred_dfs: dict[str, pd.DataFrame],
    total_frames: dict[str, int],
    title: str = "GT vs predicted detections per frame",
) -> plt.Figure:
    """Per-frame detection counts (GT green, pred blue) for all sequences."""
    n = len(gt_dfs)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, seq in zip(axes, gt_dfs):
        tf    = total_frames[seq]
        fn_idx = pd.Series(range(1, tf + 1))

        gt_fc   = gt_dfs[seq].groupby("frame_norm").size().reindex(fn_idx, fill_value=0)
        pred_fc = pred_dfs[seq].groupby("frame_norm").size().reindex(fn_idx, fill_value=0)

        sm = 9  # smoothing window
        gt_sm   = gt_fc.rolling(sm,   center=True, min_periods=1).mean()
        pred_sm = pred_fc.rolling(sm, center=True, min_periods=1).mean()

        ax.fill_between(fn_idx, gt_sm,   alpha=0.2, color="#2E7D32")
        ax.fill_between(fn_idx, pred_sm, alpha=0.2, color="#1565C0")
        ax.plot(fn_idx, gt_sm,   lw=1.1, color="#2E7D32", label="GT")
        ax.plot(fn_idx, pred_sm, lw=1.1, color="#1565C0", label="Pred", ls="--")
        ax.set_title(seq[-8:], fontsize=9, loc="left")
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frame number")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_id_switches(
    accs: dict[str, mm.MOTAccumulator],
    total_frames: dict[str, int],
    title: str = "ID switches over time",
) -> plt.Figure:
    """Plot cumulative ID switches per frame for all sequences."""
    n = len(accs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, seq in zip(axes, accs):
        ev = accs[seq].events
        if ev is None or ev.empty:
            ax.set_title(seq[-8:]); continue

        switches = ev[ev["Type"] == "SWITCH"]
        if switches.empty:
            ax.text(0.5, 0.5, "0 ID switches", ha="center", transform=ax.transAxes)
        else:
            # Cumulative count over frames
            sw_frames = switches.index.get_level_values("FrameId")
            cumsum = pd.Series(1, index=sw_frames).sort_index().cumsum()
            cumsum = cumsum.reindex(range(1, total_frames[seq] + 1), method="ffill").fillna(0)
            ax.step(cumsum.index, cumsum.values, where="post", color="#FF9800", lw=1.5)
            ax.fill_between(cumsum.index, cumsum.values, alpha=0.2, color="#FF9800", step="post")

        ax.set_title(seq[-8:], fontsize=9)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Cumulative IDSW")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

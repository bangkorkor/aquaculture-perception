# This script generates scatter plots comparing object detection models on various datasets,
# showing the trade-off between speed (FPS) and accuracy (mAP50 or mAP50-95). 
# Each point represents a model, with bubble size indicating the number of parameters. 
# The plots are saved as PNG images in the specified output directory.

import os
import argparse

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import pandas as pd
import matplotlib.pyplot as plt


DATASETS = {
    "ruod": [
        {"Model": "YOLOv8n",      "Family": "Ultralytics", "mAP50": 76.7, "mAP50_95": 55.1, "Params_M": 3.0,  "GFLOPs": 8.1,   "FPS": 139.0, "Inference_ms": 5.5,   "Preproc_ms": 1.7},
        {"Model": "YOLOv8s",      "Family": "Ultralytics", "mAP50": 79.5, "mAP50_95": 59.8, "Params_M": 11.1, "GFLOPs": 28.5,  "FPS": 141.0, "Inference_ms": 5.4,   "Preproc_ms": 1.7},
        {"Model": "YOLOv11n",     "Family": "Ultralytics", "mAP50": 77.7, "mAP50_95": 56.2, "Params_M": 2.6,  "GFLOPs": 6.3,   "FPS": 116.0, "Inference_ms": 6.9,   "Preproc_ms": 1.7},
        {"Model": "YOLOv11s",     "Family": "Ultralytics", "mAP50": 79.6, "mAP50_95": 60.0, "Params_M": 9.4,  "GFLOPs": 21.3,  "FPS": 113.0, "Inference_ms": 7.0,   "Preproc_ms": 1.8},
        {"Model": "YOLOv26n",     "Family": "Ultralytics", "mAP50": 78.5, "mAP50_95": 58.2, "Params_M": 2.4,  "GFLOPs": 5.2,   "FPS": 106.0, "Inference_ms": 8.2,   "Preproc_ms": 1.2},
        {"Model": "YOLOv26s",     "Family": "Ultralytics", "mAP50": 79.3, "mAP50_95": 60.7, "Params_M": 9.5,  "GFLOPs": 20.5,  "FPS": 102.0, "Inference_ms": 8.5,   "Preproc_ms": 1.3},
        {"Model": "RT-DETR-l",    "Family": "Ultralytics", "mAP50": 85.9, "mAP50_95": 65.9, "Params_M": 32.0, "GFLOPs": 103.5, "FPS": 39.0,  "Inference_ms": 23.6,  "Preproc_ms": 1.7},
        {"Model": "UW-YOLOv8s",   "Family": "Ultralytics", "mAP50": 81.2, "mAP50_95": 58.3, "Params_M": 8.3,  "GFLOPs": 23.4,  "FPS": 121.0, "Inference_ms": 7.6,   "Preproc_ms": 0.7},
        {"Model": "Faster RCNN",  "Family": "MMDetection", "mAP50": 77.2, "mAP50_95": 48.7, "Params_M": 41.4, "GFLOPs": 60.1,  "FPS": 6.8,   "Inference_ms": 138.9, "Preproc_ms": 8.9},
        {"Model": "Dynamic RCNN", "Family": "MMDetection", "mAP50": 78.9, "mAP50_95": 52.7, "Params_M": 41.4, "GFLOPs": 60.1,  "FPS": 6.8,   "Inference_ms": 139.1, "Preproc_ms": 8.6},
        {"Model": "DETR",         "Family": "MMDetection", "mAP50": 76.6, "mAP50_95": 48.4, "Params_M": 41.6, "GFLOPs": 22.7,  "FPS": 6.9,   "Inference_ms": 138.5, "Preproc_ms": 6.5},
        {"Model": "DINO",         "Family": "MMDetection", "mAP50": 83.5, "mAP50_95": 57.9, "Params_M": 73.2, "GFLOPs": 47.6,  "FPS": 6.5,   "Inference_ms": 150.7, "Preproc_ms": 2.6},
    ],
    "solaqua_fish": [
        {"Model": "YOLOv8n",      "Family": "Ultralytics", "mAP50": 88.1, "mAP50_95": 66.4, "Params_M": 3.0,  "GFLOPs": 8.1,   "FPS": 143.0, "Inference_ms": 5.4,   "Preproc_ms": 1.6},
        {"Model": "YOLOv8s",      "Family": "Ultralytics", "mAP50": 86.8, "mAP50_95": 68.8, "Params_M": 11.1, "GFLOPs": 28.4,  "FPS": 143.0, "Inference_ms": 5.4,   "Preproc_ms": 1.6},
        {"Model": "YOLOv11n",     "Family": "Ultralytics", "mAP50": 84.7, "mAP50_95": 62.5, "Params_M": 2.6,  "GFLOPs": 6.3,   "FPS": 119.0, "Inference_ms": 6.9,   "Preproc_ms": 1.5},
        {"Model": "YOLOv11s",     "Family": "Ultralytics", "mAP50": 86.3, "mAP50_95": 66.8, "Params_M": 9.4,  "GFLOPs": 21.3,  "FPS": 116.0, "Inference_ms": 7.0,   "Preproc_ms": 1.6},
        {"Model": "YOLOv26n",     "Family": "Ultralytics", "mAP50": 83.4, "mAP50_95": 63.0, "Params_M": 2.4,  "GFLOPs": 5.2,   "FPS": 106.0, "Inference_ms": 8.4,   "Preproc_ms": 1.0},
        {"Model": "YOLOv26s",     "Family": "Ultralytics", "mAP50": 87.2, "mAP50_95": 68.8, "Params_M": 9.5,  "GFLOPs": 20.5,  "FPS": 105.0, "Inference_ms": 8.5,   "Preproc_ms": 1.0},
        {"Model": "RT-DETR-l",    "Family": "Ultralytics", "mAP50": 93.0, "mAP50_95": 75.2, "Params_M": 32.0, "GFLOPs": 103.4, "FPS": 41.0,  "Inference_ms": 23.1,  "Preproc_ms": 1.4},
        {"Model": "UW-YOLOv8s",   "Family": "Ultralytics", "mAP50": 87.1, "mAP50_95": 55.8, "Params_M": 8.3,  "GFLOPs": 23.4,  "FPS": 122.0, "Inference_ms": 7.0,   "Preproc_ms": 1.2},
        {"Model": "Faster RCNN",  "Family": "MMDetection", "mAP50": 87.5, "mAP50_95": 58.0, "Params_M": 41.3, "GFLOPs": 60.1,  "FPS": 6.3,   "Inference_ms": 154.3, "Preproc_ms": 4.1},
        {"Model": "Dynamic RCNN", "Family": "MMDetection", "mAP50": 89.6, "mAP50_95": 66.0, "Params_M": 41.3, "GFLOPs": 60.1,  "FPS": 6.4,   "Inference_ms": 149.9, "Preproc_ms": 5.3},
        {"Model": "DETR",         "Family": "MMDetection", "mAP50": 90.8, "mAP50_95": 62.2, "Params_M": 41.6, "GFLOPs": 22.7,  "FPS": 6.4,   "Inference_ms": 151.6, "Preproc_ms": 4.0},
        {"Model": "DINO",         "Family": "MMDetection", "mAP50": 93.6, "mAP50_95": 73.5, "Params_M": 47.5, "GFLOPs": 73.2,  "FPS": 5.9,   "Inference_ms": 167.0, "Preproc_ms": 2.6},
    ],
    "uatd": [
        {"Model": "YOLOv8n",      "Family": "Ultralytics", "mAP50": 82.0, "mAP50_95": 38.7, "Params_M": 3.0,  "GFLOPs": 8.1,   "FPS": 147.0, "Inference_ms": 5.2,   "Preproc_ms": 1.6},
        {"Model": "YOLOv8s",      "Family": "Ultralytics", "mAP50": 84.4, "mAP50_95": 42.4, "Params_M": 11.1, "GFLOPs": 28.5,  "FPS": 137.0, "Inference_ms": 5.5,   "Preproc_ms": 1.8},
        {"Model": "YOLOv11n",     "Family": "Ultralytics", "mAP50": 80.1, "mAP50_95": 38.2, "Params_M": 2.6,  "GFLOPs": 6.3,   "FPS": 116.0, "Inference_ms": 6.9,   "Preproc_ms": 1.7},
        {"Model": "YOLOv11s",     "Family": "Ultralytics", "mAP50": 80.9, "mAP50_95": 38.9, "Params_M": 9.4,  "GFLOPs": 21.3,  "FPS": 112.0, "Inference_ms": 7.1,   "Preproc_ms": 1.8},
        {"Model": "YOLOv26n",     "Family": "Ultralytics", "mAP50": 82.9, "mAP50_95": 39.4, "Params_M": 2.4,  "GFLOPs": 5.2,   "FPS": 105.0, "Inference_ms": 8.3,   "Preproc_ms": 1.2},
        {"Model": "YOLOv26s",     "Family": "Ultralytics", "mAP50": 83.7, "mAP50_95": 38.9, "Params_M": 9.5,  "GFLOPs": 20.5,  "FPS": 102.0, "Inference_ms": 8.6,   "Preproc_ms": 1.2},
        {"Model": "RT-DETR-l",    "Family": "Ultralytics", "mAP50": 84.7, "mAP50_95": 39.7, "Params_M": 32.0, "GFLOPs": 103.5, "FPS": 38.0,  "Inference_ms": 24.5,  "Preproc_ms": 1.8},
        {"Model": "YOLOv11s-SDC", "Family": "Ultralytics", "mAP50": 82.0, "mAP50_95": 38.9, "Params_M": 9.6,  "GFLOPs": 28.7,  "FPS": 73.0,  "Inference_ms": 12.1,  "Preproc_ms": 1.8},
        {"Model": "Faster RCNN",  "Family": "MMDetection", "mAP50": 82.0, "mAP50_95": 34.5, "Params_M": 41.4, "GFLOPs": 75.2,  "FPS": 8.2,   "Inference_ms": 118.7, "Preproc_ms": 2.5},
        {"Model": "Dynamic RCNN", "Family": "MMDetection", "mAP50": 79.8, "mAP50_95": 35.1, "Params_M": 41.4, "GFLOPs": 75.2,  "FPS": 8.4,   "Inference_ms": 115.3, "Preproc_ms": 2.9},
        {"Model": "DETR",         "Family": "MMDetection", "mAP50": 79.0, "mAP50_95": 32.7, "Params_M": 41.6, "GFLOPs": 30.5,  "FPS": 8.5,   "Inference_ms": 114.6, "Preproc_ms": 3.2},
        {"Model": "DINO",         "Family": "MMDetection", "mAP50": 84.0, "mAP50_95": 37.1, "Params_M": 47.6, "GFLOPs": 95.6,  "FPS": 4.8,   "Inference_ms": 207.3, "Preproc_ms": 3.4},
    ],
    "net_fish_sonar": [
        {"Model": "YOLOv8n",      "Family": "Ultralytics", "mAP50": 92.5, "mAP50_95": 59.6, "Params_M": 3.0,  "GFLOPs": 8.1,   "FPS": 145.0, "Inference_ms": 5.2,   "Preproc_ms": 1.7},
        {"Model": "YOLOv8s",      "Family": "Ultralytics", "mAP50": 94.2, "mAP50_95": 61.2, "Params_M": 11.1, "GFLOPs": 28.4,  "FPS": 141.0, "Inference_ms": 5.4,   "Preproc_ms": 1.7},
        {"Model": "YOLOv11n",     "Family": "Ultralytics", "mAP50": 93.5, "mAP50_95": 59.6, "Params_M": 2.6,  "GFLOPs": 6.3,   "FPS": 119.0, "Inference_ms": 6.7,   "Preproc_ms": 1.7},
        {"Model": "YOLOv11s",     "Family": "Ultralytics", "mAP50": 91.8, "mAP50_95": 61.1, "Params_M": 9.4,  "GFLOPs": 21.3,  "FPS": 114.0, "Inference_ms": 7.0,   "Preproc_ms": 1.8},
        {"Model": "YOLOv26n",     "Family": "Ultralytics", "mAP50": 86.2, "mAP50_95": 56.7, "Params_M": 2.4,  "GFLOPs": 5.2,   "FPS": 108.0, "Inference_ms": 8.1,   "Preproc_ms": 1.2},
        {"Model": "YOLOv26s",     "Family": "Ultralytics", "mAP50": 93.5, "mAP50_95": 62.8, "Params_M": 9.5,  "GFLOPs": 20.5,  "FPS": 103.0, "Inference_ms": 8.4,   "Preproc_ms": 1.3},
        {"Model": "RT-DETR-l",    "Family": "Ultralytics", "mAP50": 91.8, "mAP50_95": 62.2, "Params_M": 32.0, "GFLOPs": 103.4, "FPS": 41.0,  "Inference_ms": 22.9,  "Preproc_ms": 1.7},
        {"Model": "YOLOv11s-SDC", "Family": "Ultralytics", "mAP50": 89.3, "mAP50_95": 56.1, "Params_M": 9.6,  "GFLOPs": 28.7,  "FPS": 73.0,  "Inference_ms": 12.4,  "Preproc_ms": 1.4},
        {"Model": "Faster RCNN",  "Family": "MMDetection", "mAP50": 93.5, "mAP50_95": 54.8, "Params_M": 41.4, "GFLOPs": 60.1,  "FPS": 7.0,   "Inference_ms": 138.1, "Preproc_ms": 4.4},
        {"Model": "Dynamic RCNN", "Family": "MMDetection", "mAP50": 92.9, "mAP50_95": 56.3, "Params_M": 41.4, "GFLOPs": 60.1,  "FPS": 6.5,   "Inference_ms": 147.0, "Preproc_ms": 5.7},
        {"Model": "DETR",         "Family": "MMDetection", "mAP50": 80.5, "mAP50_95": 46.9, "Params_M": 41.6, "GFLOPs": 23.4,  "FPS": 6.9,   "Inference_ms": 139.9, "Preproc_ms": 5.5},
        {"Model": "DINO",         "Family": "MMDetection", "mAP50": 96.7, "mAP50_95": 61.1, "Params_M": 47.5, "GFLOPs": 75.6,  "FPS": 6.5,   "Inference_ms": 151.7, "Preproc_ms": 2.1},
    ],
}


METRIC_CONFIG = {
    "map50": {
        "col": "mAP50",
        "ylabel": "mAP50 (%) (higher is better)",
        "suffix": "map50",
    },
    "map50_95": {
        "col": "mAP50_95",
        "ylabel": "mAP50-95 (%) (higher is better)",
        "suffix": "map50_95",
    },
}


def pretty_dataset_name(name: str) -> str:
    return name.replace("_", " ")


def make_plot(df: pd.DataFrame, dataset_name: str, metric: str, out_dir: str) -> str:
    cfg = METRIC_CONFIG[metric]
    ycol = cfg["col"]

    fig, ax = plt.subplots(figsize=(11, 7))

    for family, group in df.groupby("Family"):
        bubble_sizes = group["Params_M"] * 18 + 40
        ax.scatter(
            group["FPS"],
            group[ycol],
            s=bubble_sizes,
            alpha=0.75,
            label=family,
        )

    for _, row in df.iterrows():
        ax.annotate(
            row["Model"],
            (row["FPS"], row[ycol]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title(
        f"Model Trade-off on {pretty_dataset_name(dataset_name)}: Speed vs {cfg['suffix']}\n"
        "Bubble size = Parameters (M)"
    )
    ax.set_xlabel("FPS (higher is better)")
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, alpha=0.3)
    ax.legend(title="Family")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)

    out_name = f"model_tradeoff_{dataset_name}_{cfg['suffix']}.png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["ruod", "solaqua_fish", "uatd", "net_fish_sonar", "all"],
        help="Which dataset to plot.",
    )
    parser.add_argument(
        "--metric",
        choices=["map50", "map50_95", "both"],
        default="both",
        help="Which metric to plot.",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/plots",
        help="Directory to save plot images.",
    )
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    metrics = ["map50", "map50_95"] if args.metric == "both" else [args.metric]

    for dataset_name in datasets:
        df = pd.DataFrame(DATASETS[dataset_name])
        for metric in metrics:
            out_path = make_plot(df, dataset_name, metric, args.out_dir)
            print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
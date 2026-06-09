# Perception for Aquaculture Net Pens: Object Detection and Multi-Object Tracking

**Master's Thesis — Henrik Bang-Olsen**

This repository contains all practical work associated with a master's thesis on **object detection and multi-object tracking (MOT) in aquaculture net pen environments**, using both vision (monocular camera) and sonar (Ping 360 / multibeam) sensor data collected from a remotely operated vehicle (ROV).

A central contribution is the **SolAqua annotated dataset** — a multi-modal dataset of fish and net observations recorded inside a full-scale salmon pen — which is hosted separately at:

> **[https://github.com/bangkorkor/solaqua-annotated](https://github.com/bangkorkor/solaqua-annotated)**

---

## Table of Contents

1. [Research Context](#1-research-context)
2. [Repository Structure](#2-repository-structure)
3. [Datasets](#3-datasets)
4. [Object Detection](#4-object-detection)
5. [Multi-Object Tracking](#5-multi-object-tracking)
6. [Segmentation](#6-segmentation)
7. [Thesis Figures and Analysis Utilities](#7-thesis-figures-and-analysis-utilities)
8. [Environment Setup](#8-environment-setup)
9. [Getting Started](#9-getting-started)
10. [Suggested Experiment Workflow](#10-suggested-experiment-workflow)
11. [Outputs and Results](#11-outputs-and-results)
12. [Citation and Acknowledgements](#12-citation-and-acknowledgements)
13. [License](#14-license)

---

## 1. Research Context

Aquaculture net pens present challenging perception conditions: variable lighting, backscatter, biofouling, turbid water, and dense fish schools. This thesis investigates whether modern object detection and tracking pipelines — trained on purpose-built datasets collected at a commercial salmon farm — can reliably detect and track fish and net structures from ROV-mounted cameras and sonars.

The raw data were recorded during sea trials at a full-scale fish farm on **20 August 2024** using a ROV equipped with:
- Monocular and stereo cameras
- Ping 360 scanning sonar and multibeam sonar
- IMU, DVL, USBL, depth/pressure/temperature sensors

Environmental conditions: ~14 °C, wind 6 m/s, current 0.04–0.2 m/s, rain.
Fish cage: 50 m diameter, ~188 000 fish, ~3 000 g average weight, 27.5 mm net mesh with partial biofouling.

---

## 2. Repository Structure

Most of the analysis and processing work lives in **Jupyter notebooks** — these are the primary way to explore, process, and visualise data. Python scripts are used for training runs and longer batch jobs that are not suited to a notebook.

The most important top-level folders are:

- **`data-processing/`** — All dataset preparation pipelines, organised by modality (`sonar/`, `vision/`) and dataset. Each subdirectory contains Jupyter notebooks that walk through extraction from ROS bags, annotation import, dataset splitting, and statistics. Raw ROS bag files are in `solaqua_bags/`. Shared utilities (dataset summary, annotation visualisation) are in `utils/`.

- **`object-detection/`** — Ultralytics-based detection experiments. The entry point is `train.py`, which reads run configurations from `runs.csv`. Custom model YAML files are in `configs/models/`. Evaluation notebooks (`evaluate.ipynb`, `gt_vs_pred_*.ipynb`) are included alongside the training code.

- **`mmdetection/`** — Fork of the MMDetection library for Faster R-CNN, Dynamic R-CNN, DETR, and DINO experiments. Model configs specific to this project are in `mmdetection/configs/ruod/`, `/uatd/`, `/solaqua_fish/`, and `/net_fish_sonar/`.

- **`tracking/`** — Multi-object tracking experiments using Ultralytics-integrated trackers. Python scripts handle inference and MOT11-format export; Jupyter notebooks handle evaluation and plotting. Results are stored in `tracking/outputs/`.

- **`segmentation/`** — Exploratory segmentation experiments with FastSAM and SAM3. Inference scripts are in `src/`; outputs in `runs/`.

- **`runs/`** — All experiment outputs (trained weights, logs, metrics). Organised as `runs/mmdet/<dataset>/<config>/` for MMDetection and `object-detection/outputs/training/<dataset>/<run>/` for Ultralytics.

- **`thesis_figures/`** — Notebooks used to produce figures and tables for the thesis report.

- **`docs/`** — Documentation files: sea trial metadata (`solaqua_dataset_description.md`) and detection dataset split assignments (`detection_dataset_splits.md`).

- **`ultralytics/`** — Fork of the Ultralytics library, with custom blocks added for new model architectures.

- **`YOLOv11-SDC-main/`** — Fork of the YOLOv11-SDC sonar detector. Run under `.venv`.

- **`yolov8_fasternet-main/`** — Fork of the YOLOv8/FasterNet implementation. Run under `.venv-mmdet`.

---

## 3. Datasets

### 3.1 Overview

| Dataset | Modality | Classes | Images | Instances | Source |
|---------|----------|---------|--------|-----------|--------|
| `solaqua_fish` | Vision | `fish` | 793 | 3 153 | This work |
| `net_fish_sonar` | Sonar | `fish`, `net` | ~2 000 | ~3 000 | This work |
| `net_fish_sonar_improved` | Sonar | `fish`, `net` | 2 136 | 3 079 | This work (improved) |
| UATD | Sonar | 10 classes | — | — | Public benchmark |
| RUOD | Vision | 10 classes | — | — | Public benchmark |

The annotated SolAqua datasets (sonar and vision) are hosted at:
**[https://github.com/bangkorkor/solaqua-annotated](https://github.com/bangkorkor/solaqua-annotated)**

### 3.2 Raw Data (`data-processing/solaqua_bags/`)

Raw recordings are stored as **ROS bag files** (`.bag`) captured during net-following trials on 2024-08-20. Each bag is named by timestamp. Both a `_data.bag` (sensor/navigation data) and `_video.bag` (camera frames) are available for each recording session. Full metadata, including ROV depth, distance to net, velocity, and heading, is documented in [`docs/solaqua_dataset_description.md`](docs/solaqua_dataset_description.md).

### 3.3 Vision Dataset — `solaqua_fish`

- **Modality:** Monocular camera (RGB)
- **Class:** `fish` (single class)
- **Processing notebook:** `data-processing/vision/solaqua_fish/solaqua_fish.ipynb`
- **YAML config:** `data-processing/vision/solaqua_fish/solaqua_fish.yaml`
- **Split strategy:** Bag-disjoint (no temporal leakage between splits)

| Split | Bags | Images | Instances |
|-------|------|--------|-----------|
| train | 7 | 505 | 2 101 |
| val | 2 | 124 | 465 |
| test | 3 | 164 | 587 |
| **total** | **12** | **793** | **3 153** |

### 3.4 Sonar Dataset — `net_fish_sonar` / `net_fish_sonar_improved`

- **Modality:** Ping 360 / multibeam sonar (rasterised to 2D images)
- **Classes:** `fish`, `net`
- **Processing notebook:** `data-processing/sonar/net_fish_sonar_improved/net_fish_sonar_improved.ipynb`
- **YAML config:** `data-processing/sonar/net_fish_sonar_improved/net_fish_sonar_improved.yaml`

`net_fish_sonar_improved` is a refined version of `net_fish_sonar`: one bag's manual annotations are replaced by frames derived directly from the MOT ground-truth to improve annotation density and consistency.

| Split | Bags | Images | Instances |
|-------|------|--------|-----------|
| train | 6 | 1 336 | 1 973 |
| val | 2 | 400 | 541 |
| test | 2 | 400 | 565 |
| **total** | **10** | **2 136** | **3 079** |

### 3.5 Public Benchmarks

| Dataset | Modality | Classes | Notes |
|---------|----------|---------|-------|
| **UATD** | Sonar | 10 (human-body, ball, circle-cage, square-cage, tyre, metal-bucket, cube, cylinder, plane, rov) | Processed in `data-processing/sonar/UATD/` |
| **RUOD** | Vision | 10 (holothurian, echinus, scallop, starfish, fish, coral, diver, cuttlefish, turtle, jellyfish) | Processed in `data-processing/vision/RUOD/` |

Full split information is documented in [`docs/detection_dataset_splits.md`](docs/detection_dataset_splits.md).

### 3.6 MOT Sequences

Multi-object tracking evaluation is performed on separate sequences that are (as far as possible) disjoint from the detection training split. Sequences are identified by bag timestamp (e.g. `2024-08-20_17-34-52`). Ground-truth MOT annotations are provided in the [solaqua-annotated](https://github.com/bangkorkor/solaqua-annotated) repository.

---

## 4. Object Detection

Two complementary frameworks are used for object detection experiments.

### 4.1 MMDetection-Based Experiments (`mmdetection/`)

A fork of the [MMDetection](https://github.com/open-mmlab/mmdetection) library. Experiments are run under the `.venv-mmdet` environment.

**Models evaluated:**

| Model | Config family |
|-------|---------------|
| Faster R-CNN (R50-FPN) | `faster_rcnn_r50_fpn_*` |
| Dynamic R-CNN (R50-FPN) | `dynamic_rcnn_r50_fpn_*` |
| DETR (R50) | `detr_r50_*` |
| DINO 4-scale (R50) | `dino_4scale_r50_*` |

**Datasets with MMDetection configs:**

| Dataset | Config directory |
|---------|-----------------|
| RUOD | `mmdetection/configs/ruod/` |
| UATD | `mmdetection/configs/uatd/` |
| `solaqua_fish` | `mmdetection/configs/solaqua_fish/` |
| `net_fish_sonar` | `mmdetection/configs/net_fish_sonar/` |

All configs follow a `_base_` pattern for shared settings. Trained weights are stored under `runs/mmdet/<dataset>/<config_name>/`.

### 4.2 Ultralytics-Based Experiments (`object-detection/`)

Custom training harness built on the [Ultralytics](https://github.com/ultralytics/ultralytics) library (forked at `ultralytics/`). Experiments are run under the `.venv` environment.

**Entry point:**
```bash
cd object-detection
python train.py --id <run_id>
```
Run IDs are defined in `object-detection/runs.csv`, which maps each ID to a model config, dataset YAML, hyperparameter set, and output path.

**Models evaluated (from `runs.csv`):**

| Model family | Variants | Notes |
|---|---|---|
| YOLOv8 | n, s, m | Ultralytics baseline |
| YOLOv11 | n, s, m | Ultralytics v11 |
| YOLOv26 | n, s, m | Ultralytics v2.6 |
| RT-DETR-L | — | Transformer detector |
| UW-YOLOv8 | v1, v2, v3 | Underwater-adapted YOLO (custom architecture) |
| UODN | v1, v2, v3 | Custom underwater object detection network |
| AGW-YOLOv8 | s, m | Attention-guided width variant |
| MAS-YOLOv11 | n, s | Multi-scale attention variant |
| AquaYOLO | n, s, m | Custom sonar-adapted YOLO |

Custom model architectures are defined as YAML files in `object-detection/configs/models/`. New blocks are implemented inside the `ultralytics/` fork — see `ultralytics/ultralytics/nn/modules/block.py` and `ultralytics/ultralytics/nn/modules/__init__.py`.

**Also in this folder:**

- `YOLOv11-SDC-main/` — Fork of the [YOLOv11-SDC](https://github.com/TODO) sonar detector. Run under `.venv`.
- `yolov8_fasternet-main/` — Fork of the YOLOv8/FasterNet implementation. Run under `.venv-mmdet`.

---

## 5. Multi-Object Tracking

Tracking experiments use the Ultralytics-integrated trackers, driven by the scripts in `tracking/`. The `.venv` environment is used.

**Detectors used for tracking:**
- **Vision:** `RT-DETR-L` trained on `solaqua_fish` (`rtdetr_fish_improved_f`)
- **Sonar:** `YOLOv26s` trained on `net_fish_sonar_improved` (`yolov26s_sonar_improved_c`)

**Trackers evaluated:**
- **ByteTrack** — detection-based multi-object tracker
- **BoT-SORT** — Re-ID augmented multi-object tracker

**Output formats:**
- MOT11 `.txt` files — standard MOT evaluation format
- CVAT-compatible `.zip` archives — for annotation review
- Rendered `.mp4` videos — qualitative inspection

**Key scripts:**

| Script | Description |
|--------|-------------|
| `tracking-inference_improved_MOT11.py` | Main inference → MOT11 export (ByteTrack) |
| `tracking-inference_track_MOT11.py` | Alternative MOT11 export |
| `tracking-inference_CVAT.py` | Export for CVAT review |
| `render_inference_mp4.py` | Render tracked video from MOT11 files |
| `evaluate_vision.ipynb` | Compute MOT metrics for vision sequences |
| `evaluate_sonar.ipynb` | Compute MOT metrics for sonar sequences |

Tracking outputs (MOT11 files, MP4 demos, metric bar charts) are stored in `tracking/outputs/`.

---

## 6. Segmentation

The `segmentation/` folder contains exploratory segmentation experiments applied to fish imagery. The focus is zero-shot / promptable segmentation rather than supervised training.

**Models used:**
- **FastSAM-s** and **FastSAM-x** — Fast Segment Anything (small and extra-large)
- **SAM3** — Segment Anything Model variant

Pre-trained weights are stored in `segmentation/models/`. Inference scripts are in `segmentation/src/` (e.g. `FastSAM.py`, `sam3.py`, `sam3video.py`, `sam3_batch_detect.py`). Results (per-image and video outputs) are stored in `segmentation/runs/`.

---

## 7. Thesis Figures and Analysis Utilities

`thesis_figures/` contains notebooks and scripts used to produce the figures and tables in the thesis report:

| File | Content |
|------|---------|
| `thesis_figures/thesis_dataset_plots.ipynb` | Dataset statistics, distribution plots, and other figures used in the report |

Data-processing utilities shared across notebooks are in `data-processing/utils/`:
- `yolo_dataset_summary.py` — compute per-class statistics for YOLO-format datasets
- `yolo_viz.py` — visualise YOLO bounding box annotations on images

---

## 8. Environment Setup

Two separate Python 3.9 virtual environments are used. Keep them separate — the dependency sets are incompatible.

### 8.1 `.venv` — Ultralytics / Object Detection / Tracking

Used for: `object-detection/`, `tracking/`, `segmentation/`, `YOLOv11-SDC-main/`, and most notebooks.

```bash
# Create the environment (from repo root)
python3.9 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools
python -m pip install -U pip wheel setuptools

# Install the Ultralytics fork in editable mode
python -m pip install -e ultralytics
```

No locked requirements file is committed for this environment. The recommended approach is **ad-hoc installation**: run your script or notebook, read the `ModuleNotFoundError`, and install the missing package with `pip install <package>`. Repeat until the environment is satisfied. This is sometimes called *install-on-error* or *incremental dependency resolution*. Common packages you are likely to need include `opencv-python`, `scipy`, `matplotlib`, `pandas`, `jupyter`, and `motmetrics`.

If you encounter `ImportError: cannot import name 'ultralytics'`, set `PYTHONPATH` from repo root:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

Or use the absolute path:
```bash
export PYTHONPATH=/path/to/aquaculture-perception:$PYTHONPATH
```

### 8.2 `.venv-mmdet` — MMDetection / YOLOv8-FasterNet

Used for: `mmdetection/` and `yolov8_fasternet-main/`.

Requires CUDA 12.1 (loaded via `module load CUDA/12.8.0` on the HPC cluster).

```bash
# Create the environment
/usr/bin/python -m venv .venv-mmdet
source .venv-mmdet/bin/activate
module load CUDA/12.8.0

python -m pip install -U pip wheel setuptools

# Remove any conflicting packages
python -m pip uninstall -y torch torchvision torchaudio mmcv mmcv-lite mmengine mmdet

# Install PyTorch for CUDA 12.1
python -m pip install \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Install MMDetection dependencies
python -m pip install -U openmim
python -m pip install mmengine==0.10.7
python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# Install MMDetection from the forked repo
cd mmdetection
python -m pip install -e .
cd ..
```

---

## 9. Getting Started

### 9.1 Clone the Repository

```bash
git clone https://github.com/bangkorkor/aquaculture-perception.git
cd aquaculture-perception
```

### 9.2 Set Up Environments

Follow [Section 8](#8-environment-setup) to create `.venv` and/or `.venv-mmdet`.

### 9.3 Prepare Datasets

Dataset YAML files reference absolute paths under `/workspace/aquaculture-perception/data-processing/...`. You will need to either:
- Place the processed dataset files at the paths expected in the YAML configs, **or**
- Edit the YAML files (e.g. `data-processing/vision/solaqua_fish/solaqua_fish.yaml`) to point to your local dataset paths.

The annotated dataset is available at **[https://github.com/bangkorkor/solaqua-annotated](https://github.com/bangkorkor/solaqua-annotated)**.

Processed YOLO-format datasets should be placed at the paths defined in each `.yaml` config file, e.g.:
```
data-processing/sonar/net_fish_sonar/processed/split_yolo_fish_net/images/{train,val,test}
data-processing/vision/solaqua_fish/processed/split_yolo/images/{train,val,test}
```

Public benchmarks (UATD, RUOD) need to be downloaded separately and processed using the notebooks in `data-processing/sonar/UATD/` and `data-processing/vision/RUOD/`.

### 9.4 Run a Basic Ultralytics Object Detection Experiment

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
cd object-detection

# List available run IDs by inspecting runs.csv
# Then launch a training run:
python train.py --id yolov8s_RUOD_120e_fair

# Dry-run (print resolved config without training):
python train.py --id yolov8s_RUOD_120e_fair --dry
```

Outputs (weights, logs, metrics) are saved to the `project`/`name` directory defined in `runs.csv`, typically under `object-detection/outputs/training/`.

### 9.5 Run an MMDetection Experiment

```bash
source .venv-mmdet/bin/activate
module load CUDA/12.8.0
cd mmdetection

# Inspect resolved config
python tools/misc/print_config.py \
  configs/ruod/detr_r50_ruod_120e_pretrained.py

# Train
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/ruod/detr_r50_ruod_120e_pretrained.py

# Test with a saved checkpoint
python tools/test.py \
  configs/ruod/detr_r50_ruod_120e_pretrained.py \
  ../runs/mmdet/ruod/detr_r50_ruod_120e_pretrained/best_coco_bbox_mAP_epoch_62.pth

# Video inference
python demo/video_demo.py \
  demo/your_video.mp4 \
  configs/solaqua_fish/dino_4scale_r50_solaqua_fish_80e_pretrained.py \
  ../runs/mmdet/solaqua_fish/dino_4scale_r50_solaqua_fish_80e_pretrained/best_coco_bbox_mAP_epoch_4.pth \
  --out result.mp4
```

### 9.6 Run a Tracking Experiment

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
cd tracking

# Run tracking inference on a vision sequence (produces MOT11 .txt output)
python tracking-inference_improved_MOT11.py

# Evaluate results
jupyter notebook evaluate_vision.ipynb
```

Edit the `MODEL_PATH` and `SOURCE` variables at the top of the inference scripts to point to your trained weights and image sequence.

---

## 10. Suggested Experiment Workflow

```
1. Raw data (ROS bags)
        │
        ▼
2. Extract frames
   data-processing/sonar/SOLAQUA/raw_processing.ipynb
   data-processing/vision/SOLAQUA/raw_processing.ipynb
        │
        ▼
3. Annotate (external tool, e.g. Label Studio / CVAT)
   → solaqua-annotated repo
        │
        ▼
4. Build YOLO-format dataset splits
   data-processing/vision/solaqua_fish/solaqua_fish.ipynb
   data-processing/sonar/net_fish_sonar_improved/net_fish_sonar_improved.ipynb
        │
        ├──► 5a. Ultralytics detection training
        │         object-detection/train.py --id <run_id>
        │
        └──► 5b. MMDetection training
                  mmdetection/tools/train.py <config>
                        │
                        ▼
             6. Evaluate detection
                object-detection/evaluate.ipynb
                mmdetection/tools/test.py
                        │
                        ▼
             7. Tracking inference
                tracking/tracking-inference_improved_MOT11.py
                        │
                        ▼
             8. Evaluate tracking
                tracking/evaluate_vision.ipynb
                tracking/evaluate_sonar.ipynb
```

---

## 11. Outputs and Results

### Ultralytics Runs

Training outputs are written to the path defined by `project`/`name` in `runs.csv`, rooted at `object-detection/`. By default this is:

```
object-detection/outputs/training/<dataset>/<run_name>/
  weights/
    best.pt
    last.pt
  results.csv
  args.yaml
  ...
```

### MMDetection Runs

```
runs/mmdet/<dataset>/<config_name>/
  best_coco_bbox_mAP_epoch_<N>.pth
  last.pth
  <date_time>.log
  ...
```

### Tracking Outputs

```
tracking/outputs/
  inference_annotation_MOT11/      # MOT11 .txt files per sequence and tracker
  inference_annotation_MOT11_CVAT/ # CVAT-compatible annotation zips
  inference_mp4/                   # Rendered tracking videos
  labeled_mp4_demos/               # GT and inference overlay demos
  *.png                            # Metric bar charts
```

---

## 12. Citation and Acknowledgements

This repository builds on the following open-source projects. Please also cite them where appropriate:

- **MMDetection:** Chen et al., "MMDetection: Open MMLab Detection Toolbox and Benchmark." [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- **Ultralytics YOLO:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **ByteTrack:** Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV 2022.
- **BoT-SORT:** Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking." arXiv 2206.14651.
- **FastSAM:** Zhao et al., "Fast Segment Anything." arXiv 2306.12156.
- **UATD Dataset:** Xie, K., Yang, J., & Qiu, K. (2022). *A Dataset with Multibeam Forward-Looking Sonar for Underwater Object Detection.* arXiv:2212.00352.
- **RUOD Dataset:** Fu, C., et al. (2023). *Rethinking general underwater object detection: Datasets, challenges, and solutions.* Neurocomputing, vol. 517. https://doi.org/10.1016/j.neucom.2022.11.021
- **Raw SolAqua data:** SINTEF Ocean, [https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6](https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6)

---

## 14. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Third-party forks included in this repository (`mmdetection/`, `ultralytics/`, `YOLOv11-SDC-main/`, `yolov8_fasternet-main/`) retain their own licenses. Refer to the respective `LICENSE` files within each subdirectory.

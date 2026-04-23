from ultralytics import YOLO
from pathlib import Path
import zipfile
import cv2


# Load model
model = YOLO(
    "../runs/detect/outputs/training/solaqua_fish/rt_detr_solaqua_fish_120e_fair/weights/best.pt"
)

# Input video
source = "../data-processing/vision/SOLAQUA/raw_processed/mp4s/vision_raw_2024-08-20_14-31-29.mp4"

# Output paths
source_path = Path(source)
video_name = source_path.stem

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

gt_txt_path = output_dir / f"INFERENCE_GT_{video_name}.txt"
labels_txt_path = output_dir / f"INFERENCE_GT_{video_name}_labels.txt"
zip_path = output_dir / f"INFERENCE_GT_{video_name}.zip"

# Read video metadata
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {source}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

if fps <= 0:
    raise RuntimeError(f"Invalid FPS: {fps}")

# Build class names from model
if isinstance(model.names, dict):
    class_names = [model.names[i] for i in sorted(model.names.keys())]
else:
    class_names = list(model.names)

# MOT class IDs are 1-based in labels.txt order
class_id_map = {i: i + 1 for i in range(len(class_names))}

# Run tracking
results = model.track(
    source=source,
    project="/cluster/home/henrban/aquaculture-perception/runs/video_demos/predictions/solaqua_fish/tracking",
    tracker="bytetrack.yaml",
    show=False,
    save=True,
    stream=True,
    conf=0.25,
    iou=0.45,
)

mot_lines = []

# MOT frame_id is 1-based
for frame_idx, r in enumerate(results, start=1):
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        continue

    if boxes.id is None:
        continue

    xyxy = boxes.xyxy.cpu().tolist()          # absolute pixel coords
    track_ids = boxes.id.int().cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()

    for box_xyxy, track_id, class_id in zip(xyxy, track_ids, class_ids):
        x1, y1, x2, y2 = box_xyxy

        x = max(0.0, x1)
        y = max(0.0, y1)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)

        not_ignored = 1
        mot_class_id = class_id_map[int(class_id)]
        visibility = 1.0

        line = (
            f"{frame_idx},"
            f"{int(track_id)},"
            f"{x:.2f},"
            f"{y:.2f},"
            f"{w:.2f},"
            f"{h:.2f},"
            f"{not_ignored},"
            f"{mot_class_id},"
            f"{visibility:.6f}"
        )
        mot_lines.append(line)

# Write GT txt
with open(gt_txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(mot_lines))

# Write labels txt
with open(labels_txt_path, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(f"{name}\n")

# Create CVAT MOT zip:
# GT_<video_name>.zip
# └── gt/
#     ├── gt.txt
#     └── labels.txt
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(gt_txt_path, arcname="gt/gt.txt")
    zf.write(labels_txt_path, arcname="gt/labels.txt")

print(f"Saved GT txt: {gt_txt_path}")
print(f"Saved labels txt: {labels_txt_path}")
print(f"Saved CVAT zip: {zip_path}")
print(f"Frames in video: {frame_count}")
print(f"Total MOT rows written: {len(mot_lines)}")
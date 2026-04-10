from ultralytics import YOLO

# Load an official or custom model
model = YOLO("../runs/detect/outputs/training/solaqua_fish/rt_detr_solaqua_fish_120e_fair/weights/best.pt")  # Load an official Detect model



# Perform tracking with the model

results = model.track(
    source="../data-processing/vision/SOLAQUA/raw_processed/mp4s/vision_raw_2024-08-20_13-55-34.mp4",
    tracker="bytetrack.yaml",
    show=False,
    save=True,
    stream=True,
    device="cpu", # remove for gpu/automatic
)

for r in results:
    pass

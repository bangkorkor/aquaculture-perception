from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolov8n.pt")  # Load an official Detect model



# Perform tracking with the model

results = model.track(source="../data-processing/vision/FishVideo/FishVideo.mp4", show=True, tracker="bytetrack.yaml")  # with ByteTrack



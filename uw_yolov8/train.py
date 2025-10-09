from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# Swap Ultralytics' C2f for our LC2f transparently
from models.layers import LC2f
tasks.C2f = LC2f


if __name__ == "__main__":
    # Load custom model cfg
    model = YOLO("models/uw_yolov8.yaml")

    ## takes forever on mac?
    # model.train(
    #     data="data/RUOD/ruod.yaml",
    #     epochs=300,
    #     imgsz=640,          # drop to 512 if memory still tight
    #     batch=4,            # start small on Mac; increase if stable (8, 12…)
    #     device="mps",       # Apple GPU
    #     workers=2,          # low worker count to save RAM
    #     cache=False,        # avoid RAM cache on Mac
    #     optimizer="SGD",
    #     lr0=0.01,
    #     momentum=0.937,
    #     weight_decay=5e-4,
    #     amp=True,           # mixed precision on MPS
    #     patience=50,
    #     rect=True           # rectangular batches → a bit less RAM
    # )

    model.train(
        data="data/RUOD/ruod.yaml",
        epochs=150,        # start smaller; you can resume later
        imgsz=512,         # ↓ compute ~36% vs 640
        batch=4,           # try 6 or 8 if it fits
        device="mps",
        workers=2,         # keep low on Mac
        cache=False,
        rect=False,        # enables shuffle; memory should be OK at batch 4
        mosaic=0.5,        # lighter aug
        mixup=0.0,
        optimizer="SGD",
        lr0=0.01, momentum=0.937, weight_decay=5e-4,
        amp=True,
        patience=30
    )





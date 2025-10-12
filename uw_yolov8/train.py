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
        device=0,              # first GPU on the node
        workers=4,             # set ~= CPU cores you requested
        epochs=150,
        imgsz=640,
        batch=8,               # raise if VRAM allows
        amp=True,              # mixed precision
        # --- optimization ---
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        patience=50,
        # --- augmentation (safe with new API) ---
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        # --- dataloader & memory ---
        cache=True,            # set False if RAM < 32GB or NFS is slow
        rect=True,             # rectangular batches → a bit less VRAM
        # --- bookkeeping ---
        save=True, save_period=10,
        plots=False,
        seed=42
    )





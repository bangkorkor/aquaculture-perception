
from ultralytics import YOLO


if __name__ == "__main__":
    
    # Load custom model cfg
    m = YOLO("models/aquayolo.yaml")
    m.info()  # non-zero FLOPs confirms forward path OK

    m.train(
        data="data/UATD/uatd.yaml",
        epochs=300,
        imgsz=640,
        batch=64,                    # global batch; DDP splits across GPUs
        optimizer="SGD",
        lr0=0.01,
        patience=50,
        momentum=0.937,
        weight_decay=0.0005,
        device=[0, 1, 2, 3],
        workers=4,                  # ~1 worker per GPU (start low)
        # amp=False,     # <— disable AMP self-check
        # Bookkeeping
        project="runs_aquayolo",
        name="aquayolo_n_sgd300_4gpu",
        seed=0,
        # Model scale selection for your YAML's `scales:` (prevents the "assuming n" warning)
        scale="n",                    # try "s" later if you want a bit more capacity
        pretrained=False 
    )
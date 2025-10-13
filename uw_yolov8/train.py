from ultralytics import YOLO


if __name__ == "__main__":
    
    # Load custom model cfg
    model = YOLO("models/uw_yolov8.yaml").load("weights/yolov8s.pt")  # load a pretrained model 
    model.info(verbose=True)

    model.train(
        data="data/RUOD/ruod.yaml",
        epochs=300,
        imgsz=640,
        batch=64,                    # global batch; DDP splits across GPUs
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=[0, 1, 2, 3],
        workers=4,                  # ~1 worker per GPU (start low)
        project="runs_uwyolo",
        name="fasternet_sgd300_4gpu_safe",
        seed=0,
    )

   
    





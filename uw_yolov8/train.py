from ultralytics import YOLO


if __name__ == "__main__":
    
    # Load custom model cfg
    model = YOLO("models/uw_yolov8.yaml").load("weights/yolov8s.pt")  # load a pretrained model 
    model.info(verbose=True)

    model.train(
        data="data/FishDataset/fishdataset.yaml",
        epochs=150,
        patience=30,                         # early stopping
        imgsz=640,
        batch=64,                    # global batch; DDP splits across GPUs
        cache="disk", 
        optimizer="adamw",         # smoother convergence than SGD
        lr0=0.003,                 # initial learning rate
        weight_decay=0.0005,       # standard regularization
        momentum=0.937,            # used if optimizer=SGD
        cos_lr=True,   
        device=[0, 1],
        workers=4,                  # ~1 worker per GPU (start low)
        # amp=False,     # <— disable AMP self-check
        project="runs_uwyolo",
        name="fishdata_adamw150_2gpu",
        seed=0,
    )

   
    





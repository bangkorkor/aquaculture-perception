# train.py
from ultralytics import YOLO

def main():
    # Load custom model cfg
    model = YOLO("configs/models/uw_yolov8.yaml").load("weights/yolov8s.pt")  # load a pretrained model 
    model.info(verbose=True)

    model.train(
        data="../data-processing/vision/RUOD/ruod.yaml",
        epochs=300,
        imgsz=640,
        batch=32,                    
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=30,
        amp=False,     # <— disable AMP self-check
        device=[0],
        workers=2,
        project="runs_uwyolo",
        name="ruod_0873_sgd300_1gpu",
        seed=0,
    )



if __name__ == "__main__":
    main()
    
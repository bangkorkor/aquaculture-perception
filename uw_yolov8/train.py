from ultralytics import YOLO


if __name__ == "__main__":
    
    # Load custom model cfg
    model = YOLO("models/uw_yolov8.yaml")
    model.info(verbose=True)

    model.train(
        data="path/to/your_data.yaml",
        epochs=300,
        imgsz=640,
        batch=64,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=0,
    )

   
    





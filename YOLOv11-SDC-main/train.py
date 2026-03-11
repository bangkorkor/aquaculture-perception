import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11s-SDC.yaml')

    model.train(
        data='/cluster/home/henrban/aquaculture-perception/YOLOv11-SDC-main/ultralytics/cfg/datasets/UATD.yaml',

        epochs=120,
        imgsz=640,
        batch=32,
        nbs=64,

        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=3.0,
        warmup_momentum=0.8,
        cos_lr=False,

        amp=False,
        device=0,
        workers=8,
        seed=0,
        deterministic=False,
        pretrained=True,

        fliplr=0.5,
        flipud=0.0,

        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,


        rect=False,
        project='runs/SDC',
        name='uatd_120e_batch64_fair',
    )
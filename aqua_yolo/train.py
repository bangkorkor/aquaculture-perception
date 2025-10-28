
from ultralytics import YOLO


if __name__ == "__main__":
    
    # Load custom model cfg
    m = YOLO("models/aquayolo.yaml")
    m.info()  # non-zero FLOPs confirms forward path OK

    m.train(
        data="data/UATD/uatd.yaml",
        pretrained=False,          # Safest off
        epochs=300, 
        imgsz=640,
        batch=64,

    # optimizer
        optimizer="SGD",           # <- paper-style; avoids "auto" ambiguity
        lr0=0.012,
        momentum=0.937,
        weight_decay=6e-4,
        cos_lr=True,
        lrf=0.01,                  # final LR ratio for cosine
        warmup_epochs=10,
        patience=50,               


    # loss balance – tilt a bit toward recall
        box=10.0, cls=0.6, dfl=1.3,


    # sonar-friendly aug, we do nothing really, is this ok?
        auto_augment=None,  
        # geometry
        degrees=0.2,          # no rotation (shadow direction is meaningful)
        shear=0.0,
        perspective=0.0,
        translate=0.02,       # small shifts are ok
        scale=0.35,            # Ultralytics default scale jitter; fine for sonar

        # photometrics
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,  # disable HSV for gray-scale sonar

        # composition
        mosaic=0.0,           # off (mosaic creates unrealistic seabed continuity)
        mixup=0.0,            # off (blends break acoustic edges)
        copy_paste=0.0,       # off (unlikely to preserve shadows correctly)

        # flips
        fliplr=0.05,           # small chance; left/right symmetry may be acceptable
        flipud=0.0,           # avoid flipping seabed “upside-down”

        # occlusion-like
        erasing=0.15,          # light Random Erasing to mimic dropouts/occlusions
        
    

        
    # system
        workers=8,                 # A100s handle this fine
        device=[0, 1, 2, 3],
        project="runs_aquayolo",
        name="aquayolo_L_SGD_640_4gpu_fullmatch",
        seed=0,
        amp=False,  # this makes it slower but stable? Uses a lot of memory?
    )
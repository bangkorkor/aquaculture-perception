
from ultralytics import YOLO


if __name__ == "__main__":
    
    # Load custom model cfg
    m = YOLO("models/aquayolo.yaml")
    m.info()  # non-zero FLOPs confirms forward path OK

    m.train(
        data="data/CFC_gray/cfc_gray.yaml",
        pretrained=False,          # Safest off
        epochs=60, 
        imgsz=640,
        batch=512,
        nbs=512,


    # optimizer
        optimizer="AdamW",
        lr0=3e-4,
        weight_decay=0.01,
        cos_lr=True,
        lrf=0.01,
        warmup_epochs=3,
        patience=20,            


    # loss balance – tilt a bit toward recall
        box=10.0, cls=0.6, dfl=1.3,


    # sonar-friendly aug, we do nothing really, is this ok?
        auto_augment=None,  
        # geometry
        degrees=0.0,          # no rotation (shadow direction is meaningful)
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
        workers=8,                 # safe
        device=[0, 1, 2, 3],
        project="runs_aquayolo",
        name="cfc_gray_adamW_m",
        seed=0,
        plots=False,
        cache=False,
        amp=False,  
    )


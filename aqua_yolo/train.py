from ultralytics import YOLO


if __name__ == "__main__":
    
    # Load custom model cfg
    m = YOLO("models/aquayolo.yaml")
    m.info()  # non-zero FLOPs confirms forward path OK

    m.train(
        data="../solaqua/data/net_sonar/net_sonar.yaml",
        pretrained=False,          # Safest off
        epochs=150, 
        patience=50,   
        imgsz=1280,
        batch=12,
        nbs=64,

    # optimizer 
        optimizer="SGD",
        lr0=0.005,                      # conservative for SGD; adjust later if needed
        momentum=0.9,
        weight_decay=5e-4,
        cos_lr=True,
        lrf=0.01,
        warmup_epochs=20,     
                 


    # loss balance – tilt a bit toward recall ????
        box=10.0, cls=0.6, dfl=1.3,


    # augmentation, light
        rect=False,
        mosaic=0.08,
        close_mosaic=100, # only on for the 50 first. 
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=3.0, shear=0.0, perspective=0.0,
        translate=0.06,
        scale=0.20,
        fliplr=0.0, flipud=0.0,
        

        
    # system
        workers=2,                 # safe
        device=0,
        project="runs_aquayolo",
        name="net_sonar_SGD_1280_lightaug",
        seed=0,
        plots=False,
        cache=False,
        amp=False,   
    )

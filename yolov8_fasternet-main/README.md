# NOTES FROM HENRIK
Use the .venv-mmdet enviornemnt. It is the models/v8/yolov8s-fasternet-Fbifpn-GSFC2f.yaml that is the model presented in the paper. 

For training do:
python train.py \
  --preset ruod_120e_batch32_fair \
  --yaml ultralytics/models/v8/yolov8s-fasternet-Fbifpn-GSFC2f.yaml \
  --data RUOD.yaml \
  --project runsD \
  --name yolov8s-fasternet \
  --cache ram

For testing do:
python val.py --weight ./runsD/yolov8s-fasternet/weights/best.pt --data RUOD.yaml --name yolov8_fasternet


The weights are in runsD. The training is done in train.py and the evaluation is done on val.py. These files are modified to follow common setup that all training and evaluation uses. 

Remember that the dataset .yaml config files are in ultralytics/datasets/. And that the models are in ultralytics/models/v8/. All custom blocks made by the author of the original paper are in ultralytics/nn.

---


# A lightweight YOLOv8 integrating FasterNet for real‑time underwater object detection, J Real-Time Image Proc 21, 49 (2024).

Code to reproduce the experiments in the paper [A lightweight YOLOv8 integrating FasterNet for real‑time underwater object detection](https://link.springer.com/article/10.1007/s11554-024-01431-x)


## Prerequisits

1. Download and prepare the datasets.  [RUOD](at https://github.com/dlut-dimt/RUOD) from their official sources.

the structure of the dataset:
   ./datasets
    ├── RUOD
    │   ├── train        
    │   │   ├── images
    │   │   └── labels
    │   ├── test        
    │   │   ├── images
    │   │   └── labels
    │   ├── val        
    │   │   ├── images
    │   │   └── labels

modify the directory in ./ultralytics/datasets/RUOD.yaml
    train: ./datasets/RUOD/train/images
    val:   ./datasets/RUOD/val/images
    test: ./datasets/RUOD/test/images


###train
python train.py --device 0 --yaml ultralytics/models/v8/yolov8s-fasternet.yaml --data RUOD.yaml --workers 8 --batch 64 --name yolov8s-fasternet --project runsD --imgsz 640 --cache --epochs 100

###test
python val.py --weight ./runs/train/RUOD/yolov8s/weights/best.pt --data RUOD.yaml --split test  --batch 1 --project runs/val/RUOD --name yolov8s-fasternet

## Acknowledgements
This project is built upon:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Original YOLOv8 implementation
- https://github.com/JierunChen/FasterNet    -Faster Neural Networks in "Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks"  CVPR2023.

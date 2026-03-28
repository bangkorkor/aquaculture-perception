# Obstacle Detection and Avoidance for UUVs using Vision and Mutli-Beam sensor data

This is the Master's project for Henrik Bang-Olsen. This repo will contain all the code used for the project.

This README will describe the structure of the project and how to get started.




## Project Structure
- mmdetection has the mmdetection library forked. Here we run mmdetection experiments within the .venv-mmdet enviornemnt
- object-detection has the setup for the vanilla ultralytics experiments. This is my setup, the actual code for the ultralytics is found in ultralytics/ folder. this code is ran on the .venv enviornment. 
- ultralytics has the ultralytics fork and i have tried customizing blocks for adding new models. But this folder mainly has the entire ultralytics library.
- yolov8_fasternet-main has u-yolov8 training and testing. This is a fork from the authors of the original paper. This code needs to be run no the .venv-mmdet environment. All the results from this expriment is also within this folder.
- YOLOv11-SDC-main ha code for this sonar detector. Should be in the .venv enviornment. 




## Environments Setup Ultralytics

- Do 'python -m pip freeze > requirements.lock.txt' to get a .txt file of all dependencies. 
- I use pyhon3.9 or newer. To make an enviorment do 'python3.9 -m venv .venv' (or 'python3 ...' ??)
- To activate it do: 'source .venv/bin/activate'
- 'python -m pip install -U pip wheel setuptools' updates the packaging tools inside .venv
- 'python -m pip install -r requirements.lock.txt' to install all packages inside requirements.lock.txt
- 'python -m pip install -e ultralytics' to install ultralytics fork from this repo?

## Quick comand troule:
Sometimes i get that we cant import the ultralytics, this is because it cant find the init. Do this from root:
```
export PYTHONPATH=$PWD:$PYTHONPATH
```


## 🧾 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

# MMdetection

## MMdetect setup:

```bash 
# make .venv-mmdet
/usr/bin/python -m venv .venv-mmdet
source .venv-mmdet/bin/activate
module load CUDA/12.8.0

python -m pip install -U pip wheel setuptools
python -m pip uninstall -y torch torchvision torchaudio mmcv mmcv-lite mmengine mmdet

python -m pip install \
  torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -U openmim
python -m pip install mmengine==0.10.7

python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```


## How the code works
Models are setup using config files. For fair experiments with my setup we use a _base_ setup. The actual model is configed in its own .py file. These files are (for the RUOD experiments) are found in mmdetection/configs/ruod/


## commands

All these command shaould be run from mmdetection root and the .venv-mmdet environment needs to be activated. 

Activating venv and loading cuda:
```bash
source .venv-mmdet/bin/activate
module load CUDA/12.8.0
```

Seeing the setup: (change the path of the model)
```bash
python tools/misc/print_config.py \
  configs/ruod/detectors_cascade_rcnn_r50_ruod_120e_pretrained.py
```

Getting GFLOPS: (Does not work for every detector)
```bash
python tools/analysis_tools/get_flops.py \
  configs/ruod/detr_r50_ruod_120e_pretrained.py
```

Do the training:
```bash
module load CUDA/12.8.0
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/ruod/detectors_cascade_rcnn_r50_ruod_120e_pretrained.py
```

Do the testing:
```bash
python tools/test.py \
  configs/ruod/detr_r50_ruod_120e_pretrained.py \
  ../runs/mmdet/ruod/detr_r50_ruod_120e_pretrained/best_coco_bbox_mAP_epoch_62.pth
```


Run video inference: (place video in demo-folder)
```bash
python demo/video_demo.py \
    demo/vision_raw_2024-08-20_17-14-36.mp4 \
    configs/solaqua_fish/dino_4scale_r50_solaqua_fish_80e_pretrained.py \
    ../runs/mmdet/solaqua_fish/dino_4scale_r50_solaqua_fish_80e_pretrained/best_coco_bbox_mAP_epoch_4.pth \
    --out result.mp4
```

# RunPod
### Moving files in runpod with rsync
```bash
/opt/homebrew/bin/rsync -avP \
  -e "ssh -p 12211 -i ~/.ssh/id_ed25519 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes -o Compression=no" \
  UATD_Training.zip root@157.157.221.29:/workspace/
```

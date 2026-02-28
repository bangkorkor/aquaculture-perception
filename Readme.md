# Obstacle Detection and Avoidance for UUVs using Vision and Mutli-Beam sensor data

This is the Master's project for Henrik Bang-Olsen. This repo will contain all the code used for the project.

This README will describe the structure of the project and how to get started.




## Project Structure
TODO




## Environments Setup

- Do 'python -m pip freeze > requirements.lock.txt' to get a .txt file of all dependencies. 
- I use pyhon3.9 or newer. To make an enviorment do 'python3.9 -m venv .venv' (or 'python3 ...' ??)
- To activate it do: 'source .venv/bin/activate'
- 'python -m pip install -U pip wheel setuptools' updates the packaging tools inside .venv
- 'python -m pip install -r requirements.lock.txt' to install all packages inside requirements.lock.txt
- 'python -m pip install -e ultralytics' to install ultralytics fork from this repo?




## 🧾 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## MMdetect setup:

```bash 
# make .venv-mmdet
/usr/bin/python -m venv .venv-mmdet
source .venv-mmdet/bin/activate

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


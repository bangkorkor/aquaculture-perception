# Object detection using YOLO

Files:

- train.py, this is the python file we execute to run the actual training. Here everything below is put together. 
- train.json. The idea here is that this routes everything together. We have a training instance (id) that has fields to all relevant training stuff. 
- utils/, this is a folder with helper code to use in train.py to keep stuff clean.
- configs/models/, yaml files that builds the networks. The costum moduels are created in Ultralytics/.
- configs/training-params.json, is training spesific parameters that are used in model.train() function

How to run:

python train.py --id ruod_0873_sgd300_1gpu

- The initialization is set in runs.csv where the id in runs.csv should match the id that we run. 
- Make new run-id to start a new experiment. 





## Building models

### Building

Building (and training) of the model happens in [train.py](../train.py). The model is constructed by the [aquayolo.yaml](../models/aquayolo.yaml), that imports modules from custom blocks (see paragraph below).

### Adding custom blocks

I have added custom blocks to match the architecture as pruposed in the paper. For this to work I have first cloned the ultralytics repo.
- Head to [ultralytics](https://docs.ultralytics.com/guides/model-yaml-config/#source-code-modification) to see how to add custom blocks.
- Files that is modified: [block.py](../../ultralytics/ultralytics/nn/modules/block.py), [tasks.py](../../ultralytics/ultralytics/nn/tasks.py) (added imports and updated parse_model() for special arguments), [__init__.py](../../ultralytics/ultralytics/nn/modules/__init__.py) (exposing the modules).


# If it cant find utralytics
Change this to fix the paths???

- source /cluster/home/henrban/aquaculture-perception/.venv/bin/activate
- export PYTHONPATH=/cluster/home/henrban/aquaculture-perception:$PYTHONPATH




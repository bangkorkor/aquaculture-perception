# Object detection using YOLO

Files:

- train.py, this is the python file we execute to run the actual training. Here everything below is put together. 
- train.json. The idea here is that this routes everything together. We have a training instance (id) that has fields to all relevant training stuff. 
- utils/, this is a folder with helper code to use in train.py to keep stuff clean.
- configs/models/, yaml files that builds the networks. The costum moduels are created in Ultralytics/.
- configs/training-params.json, is training spesific parameters that are used in model.train() function

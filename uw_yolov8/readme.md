# UW-YOLOv8

In this folder I will replicate the model-architecture proposed in the paper: [A lightweight YOLOv8 integrating FasterNet for real-time underwater object detection](https://www.researchgate.net/publication/378873018_A_lightweight_YOLOv8_integrating_FasterNet_for_real-time_underwater_object_detection).

## Architecture


## Data

- RUOD dataset avilable at: [RUOD](https://github.com/xiaoDetection/RUOD?tab=readme-ov-file)

```bash
{
    "images": [
        {
            "id": 1,
            "file_name": "005310.jpg",
            "width": 498,
            "height": 644
        },  # ... all the images
        ],
    
    "type": "instances",
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "bbox": [
                8,
                143,
                327,
                436
            ],
            "category_id": 6,
            "area": 142572,
            "iscrowd": 0,
            "ignore": 0
        },  # ... all the annotations
        ],
     "categories": [
        {
            "id": 1,
            "name": "holothurian"
        },  # ... all the categories
        {
            "id": 10,
            "name": "jellyfish"
        }
    ]
}
```

## Running Code / Viewing Results

- Activate the myenv
- Do pip install requirements.txt
- Run through the notebook, do all the steps. 
- Rename RUOD_pic to images. 
- Update the paths for the ruod.yaml. The "path:" line should be commented out. 
- Create a __init__.py file inside models?????
- To run the actual script do PYTHONPATH=$PWD python3 train.py
- i think thats it.
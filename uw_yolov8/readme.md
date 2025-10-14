# UW-YOLOv8

In this folder I will replicate the model-architecture proposed in the paper: [A lightweight YOLOv8 integrating FasterNet for real-time underwater object detection](https://www.researchgate.net/publication/378873018_A_lightweight_YOLOv8_integrating_FasterNet_for_real-time_underwater_object_detection). 

## Getting started / View results

To see the results without running the code you can always head to the uw-yolov8.ipynb notebook. The model weights are stored inside /uw_yolov8/runs_uwyolo/fasternet_sgd300_4gpu_safe/weights/best.pt and can be downloaded if needed. 

**If you want to run the code yourself**:
- Clone the repo, you need dependencies like Python, pip, Jupyter etc.
- Setup a virtual enviornment:
   - Create the venv: `python3 -m venv myenv`
   - Activate the venv: `source myenv/bin/activate`
- Navigate to the ultralytics folder and do `pip install -e .`
- Navigate back to the uw_yolov8 folder and do `pip install -r requirements.txt`, if there are other dependencies that needs to be installed also, just fix the error messages as you go!
- Navigate to the uw-yolov8.ipynb notebook and select a kernel, choose the myenv you have created.
- This should be it for the setup. You should now follow the notebook for the rest of the runthrough. 


## Architecture
This model tries to replicate the architecture propused in the [paper](https://www.researchgate.net/publication/378873018_A_lightweight_YOLOv8_integrating_FasterNet_for_real-time_underwater_object_detection). See the [yaml] file for how it is implemented

<img width="400" height="498" alt="uw_yolov8 architecture" src="https://github.com/user-attachments/assets/ed3c0959-ad1f-4dc4-bb28-9907d2c69e48" />


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



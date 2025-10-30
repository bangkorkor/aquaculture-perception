# Obstacle Detection and Avoidance for UUVs using Vision and Mutli-Beam sensor data

This is the specialization project for Henrik Bang-Olsen. This repo will contain all the code used for the project.

This README will describe the structure of the project and how to get started.


**TODO**
- Fix folder struture and naming conventions. 
    - We shall make a new folder called notebooks (this is now utils): in these notebooks we mainly process the datasets and show the predictions. 
    - We should match the model's (the notebook and the run) name based on what data it is trained on. All the models within this root is obviuosly following the uw_yolo architecture. 
    - One model should be tested on different datasets. All this should be done in the notebook. I shall write about the resuluts in my notion-paper, keeping everything clear. 
- Write about this new folder structure in readme.md

## Project Structure
This repo contains different models/tools for underwater object detection using vision and sonar data. The different folders are listed below and they contain there own readmes with instructions. 

### 👁️ Vision models:
- [`uw_yolov8/`](uw_yolov8) - Yolov8 model with FasterNet backbone.

### 🛜 Sonar models:
- [`aqua_yolo/`](aqua_yolo) - Not complete.

### ⚙️ External:

- [`ultralytics/`](ultralytics) - Modified ultralytics fork (forked on 13.10.2025) for using custom YOLO-models.
         




## Setup

- Should create venv. 
- Setup: python3 -m venv myenv
- To activate: source myenv/bin/activate 
- To deactivate: deactivate
- Navigate to your model, see Project Structure
- Connect you notebook to a kernel and choose myenv. 


## 🧾 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


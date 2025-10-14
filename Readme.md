# Obstacle Detection and Avoidance for UUVs using Vision and Mutli-Beam sensor data

This is the specialization project for Henrik Bang-Olsen. This repo will contain all the code used for the project.

This README will describe the structure of the project and how to get started.


**TODO**
- Make root readme. Add license etc
- Complete uw_yolov8?
- Start on sonar model


## Project Structure
This repo contains different models/tools for underwater object detection using vision and sonar data. The different folders are listed below and they contain there own readmes with instructions. 

### 👁️ Vision models:
- [`uw_yolov8/`](./uw_yolov8/README.md) - Yolov8 model with FasterNet backbone.

### 🛜 Sonar models:
- [`aqua_yolo/`](./aqua_yolo/README.md) - Not complete.

### ⚙️ External:

- [`ultralytics/`](./ultralytics/README.md) - Modified ultralytics fork (forked on 13.10.2025) for using custom YOLO-models.
         




## Setup

- Should create venv. 
- Setup: python3 -m venv myenv
- To activate: source myenv/bin/activate 
- To deactivate: deactivate
- Navigate to your model, see Project Structure
- Connect you notebook to a kernel and choose myenv. 


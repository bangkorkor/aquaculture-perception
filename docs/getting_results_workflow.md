# Workflow of getting results

This md is for myself to remember how to get the results

getting data -> processing data -> yolo configuration (yaml) -> building architecture -> training -> evaluation


## Walkthrough on testing net_sonar

0. Having training dataset and setup on YOLO-format. This is done in notebook like aqua_yolo/net_sonar.ipynb e.g
1. Building model. Done in same notebook as in step0.
2. Training model. Done in train.py
3. (Optinal) Evaluate model. Done in same notebook as in step0. 
4. Visually testing on another bag. We use all the frames from a bag and create an images-folder. This is done in solaqua/solaqua_sonar_dataprocessing.ipynb. Select bag, config for images and in/out dir. Uploading all images takes some time.
    41. Do the actual predictions. This is done in aqua_yolo/solaqua_visual_evaluation.ipynb. The images predicted images should be saved in a predictions folder. This takes some time.
5. Visually testing for vision images in solaqua/solaqua_vision_dataprocessing.ipynb. This is for getting the side_by_side view later. Get all raw images from bag in uw_yolov8/solaqua_vision_dataprocessing.ipynb
    51. Do the actual predictions. This is done in uw_yolov8/solaqua_visual_evaluation.ipynb under Annotate ... The images will be saved in prediction folder. 
6. Make sbs video. This is done in solaqua/solaqua_combined_results.ipynb. DONE!

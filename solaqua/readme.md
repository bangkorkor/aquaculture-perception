# Solaqua

notebook is moved to uw_yolo/notebooks

Not important now, for later.

dataset_extraction.ipynb has code for extracting and vizualizing the solaqua dataset. 


### Datasets

All data files should be placed in the `./data` folder.

- `*_data.bag` → contains **sensor data** (ROS bag format).
- `*_video.bag` → contains **video images** (ROS bag format).

The dataset used is **SOLAQUA**, available from [SINTEF Open Data](https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6).



### Running Code

1. Place `.bag` files in `data/`
2. Run notebook:
   - `dataset_extraction.ipynb` → run all

Output files will be placed in `output/` folder


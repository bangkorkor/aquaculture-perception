# Obstacle Detection and Avoidance for UUVs using Vision and Mutli-Beam sensor data

Specialization Project for Henrik Bang-Olsen +++

---

## Setup

Should create venv. 
Setup: python3 -m venv myenv
To activate: source myenv/bin/activate 
To deactivate: deactivate

Connect you notebook to a kernel and choose myenv. 

---

## Project Structure

```bash
📂 myenv/              # User-created virtual environment (not tracked in git)

📂 model_x/            # Folder for a specific underwater detection model
 ┣ 📂 data/            # Training / testing datasets or dataset links
 ┣ 📂 utils/           # Helper scripts and functions
 ┣ 📓 notebook.ipynb   # Jupyter notebook with experiments / training code
 ┗ 📄 README.md        # Model-specific documentation

📂 model_y/
 ┣ 📂 data/
 ┣ 📂 utils/
 ┣ 📓 notebook.ipynb
 ┗ 📄 README.md

📄 README.md           # Main repository documentation
```
Each model folder is self-contained with its own dataset, utilities, notebook(s), and documentation, making it easier to experiment with and extend different approaches.

---
## Solaqua

This is under the solaqua folder. not important now, for later.

dataset_extraction.ipynb has code for extracting and vizualizing the solaqua dataset. 

### Datasets

All data files should be placed in the `./data` folder.

- `*_data.bag` → contains **sensor data** (ROS bag format).
- `*_video.bag` → contains **video images** (ROS bag format).

The dataset used is **SOLAQUA**, available from [SINTEF Open Data](https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6).

---

### Running Code

1. Place `.bag` files in `data/`
2. Run notebook:
   - `dataset_extraction.ipynb` → run all

Output files will be placed in `output/` folder

---

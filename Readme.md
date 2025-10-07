# Obstacle Detection and Avoidance for UUVs using Vision and Mutli-Beam sensor data

Specialization Project for Henrik Bang-Olsen +++



## Setup

- Should create venv. 
- Setup: python3 -m venv myenv
- To activate: source myenv/bin/activate 
- To deactivate: deactivate
- Navigate to your model, see Project Structure
- Connect you notebook to a kernel and choose myenv. 



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





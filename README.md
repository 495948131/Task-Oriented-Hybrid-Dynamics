# Task-Oriented-Hybrid-Dynamics

This repository stores the code and datasets for the **task-oriented hybrid dynamics modeling framework**, which is used in research on contact force estimation for manipulators via intrinsic sensors.

![图片描述文本 (Alt Text)](NNStructure.jpg)

## Requirments
python-3.10.4<br>
pandas-2.2.1<br>
matplotlib-3.8.1<br>
numpy-1.26.1<br>
torch-2.2.2<br>
d2l-0.15.0<br>
tensorboardX-2.6.4<br>
scipy-1.11.3

## Repository Content Overview
| File/Directory               | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `model\TaskOrientedHybridDynamics.py`                    | Model definition file: Implements the LSTM network architecture for contact force estimation |
| `main.ipynb`         | Main test file: Complete workflow for loading datasets, calling the model, and executing contact force estimation |
| `data\ContactlessDataset1.csv` | Contactless Dataset 1: Contains 2600 training samples and 200 test samples |
| `data\ContactlessDataset2.csv` | Contactless Dataset 2: Contains 2600 training samples and 200 test samples |
| `data\ContactDataset1.csv` | Contact Dataset 1: Contains 2600 training samples and 200 test samples     |


## Quick Start
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/whHuang94/Task-Oriented-Hybrid-Dynamics.git
2. Run Code
   ```bash
   python main.ipynb


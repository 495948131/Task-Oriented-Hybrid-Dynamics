# Task-Oriented-Hybrid-Dynamics

This repository stores the code and datasets for the **task-oriented hybrid dynamics modeling framework**, which is used in research on contact force estimation for manipulators via intrinsic sensors.


## Repository Content Overview
| File/Directory               | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `LSTM.py`                    | Model definition file: Implements the LSTM network architecture for contact force estimation |
| `compensateNN.ipynb`         | Main test file: Complete workflow for loading datasets, calling the model, and executing contact force estimation |
| `YangbenCeshiji38(2600).csv` | Contactless Dataset 1: Contains 2600 training samples and 200 test samples |
| `YangbenCeshiji41(2600).csv` | Contactless Dataset 2: Contains 2600 training samples and 200 test samples |
| `YangbenCeshiji42(2600).csv` | Contact Dataset 1: Contains 2600 training samples and 200 test samples     |


## Quick Start
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/495948131/Task-Oriented-Hybrid-Dynamics.git

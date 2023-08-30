[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/hanxiao0607/RootCLAM/blob/main/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhanxiao0607%2FRootCLAM%2Ftree%2Fmain&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# RootCLAM: On Root Cause Localization and Anomaly Mitigation through Causal Inference
A Pytorch implementation of [RootCLAM]().

## Configuration
- Ubuntu 20.04
- NVIDIA driver 495.29.05 
- CUDA 11.3
- Python 3.9.7
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
A virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Instructions
Clone the template project, replacing ``my-project`` with the name of the project you are creating:

        git clone https://github.com/hanxiao0607/RootCLAM.git my-project
        cd my-project

Run and test:

        python3 main.py DATASET_NAME AD_MODEL_NAME

where DATASET_NAME can be adult, loan, or donors, and AD_MODEL_NAME can be ae or deepsvdd.

Eample:

        python3 main.py adult ae

## Citation
```

```
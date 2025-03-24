[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/hanxiao0607/RootCLAM/blob/main/LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
# RootCLAM: On Root Cause Localization and Anomaly Mitigation through Causal Inference
A Pytorch implementation of [RootCLAM](https://dl.acm.org/doi/10.1145/3583780.3614995).

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
@inproceedings{10.1145/3583780.3614995,
        author = {Han, Xiao and Zhang, Lu and Wu, Yongkai and Yuan, Shuhan},
        title = {On Root Cause Localization and Anomaly Mitigation through Causal Inference},
        year = {2023},
        isbn = {9798400701245},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3583780.3614995},
        doi = {10.1145/3583780.3614995},
        abstract = {Due to a wide spectrum of applications in the real world, such as security, financial surveillance, and health risk, various deep anomaly detection models have been proposed and achieved state-of-the-art performance. However, besides being effective, in practice, the practitioners would further like to know what causes the abnormal outcome and how to further fix it. In this work, we propose RootCLAM, which aims to achieve Root Cause Localization and Anomaly Mitigation from a causal perspective. Especially, we formulate anomalies caused by external interventions on the normal causal mechanism and aim to locate the abnormal features with external interventions as root causes. After that, we further propose an anomaly mitigation approach that aims to recommend mitigation actions on abnormal features to revert the abnormal outcomes such that the counterfactuals guided by the causal mechanism are normal. Experiments on three datasets show that our approach can locate the root causes and further flip the abnormal labels.},
        booktitle = {Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
        pages = {699–708},
        numpages = {10},
        keywords = {root cause analysis, causal inference, anomaly mitigation},
        location = {Birmingham, United Kingdom},
        series = {CIKM '23}
}
```

# Heart Disease Artificial Intelligence Datathon 2021

**Baseline Code**

---

## Setup

```
pyenv install 3.7.10
pyenv virtualenv 3.7.10 hdaid2021
pyenv activate hdaid2021
pip install -r requirements.txt
```

## Run as scripts

```
python train.py --device cuda:0
```

## Run on Colaboratory

See `baseline.ipynb` file.

---

## Base Model Architectures

**DeepLabV3 + Resnet101**: Baseline

* **Paper**: [Arxiv 1706.05587](https://arxiv.org/abs/1706.05587)

* **Implementation**: [Pytorch Vision](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)

**U-Net**

* **Paper**: [Arxiv 1505.04597](https://arxiv.org/abs/1505.04597)

* **Implementation**: [models/unet.py](models/unet.py)

**Inception U-Net**

* **Paper**: [ACM 10.1145/3376922](https://dl.acm.org/doi/abs/10.1145/3376922)

* **Implementation**: [models/unet.py](models/unet.py)

**RefineNet**

* **Paper**: [Arxiv 1611.06612](https://arxiv.org/abs/1611.06612)

* **Implementation**: [models/refinenet.py](models/refinenet.py)

---

# Corruption-invariant Person Re-identification Via Coarse-to-Fine Feature Alignment

## Introduction
Corruption-invariant Person Re-identification (CIReID) aims to build robust identity correspondence across nonoverlapped cameras even when severe image corruptions occur. It is challenging as those corruptions contaminate intrinsic pedestrian characteristics and cause semantic misalignment in feature space. To address this issue, this paper proposes a coarseto-fine semantic alignment framework that learns corruptioninvariant pedestrian features for re-identification from the perspective of multi-modal feature alignment. In this framework, a Coarse-to-Fine Feature Alignment Transformer (CFAT) is introduced to extract and align features of pedestrian images with different corruptions. Specifically, the CFAT aligns features of corrupted samples to that of the corresponding clean samples in a knowledge distillation manner in the coarse alignment stage, i.e., a teacher network distils identity-related semantics from clean samples and supervises the student network learning semantic-consistent features from corrupted samples. To avoid information loss of the strict alignment, we propose to integrate a Bridge Feature Generation (BFG) module into CFAT to construct meaningful latent structures among modalities in the fine alignment stage. This enables seamless alignment of the same identity between corrupted and clean modalities, leading to better re-identification performance. To evaluate the effectiveness of the proposed method, extensive experiments are conducted on three public benchmark datasets, i.e., Market-1501, CUHK-03, and MSMT-17. The experimental results demonstrate our CFAT outputs state-of-the-arts with a large margin in various corrupted scenes.

### Method overview
![image](https://github.com/user-attachments/assets/27612791-d121-41ad-aab3-b3b8e0965ecc)

## Quick Start
### 1.Dependencies
● python=3.8.13<br>
● pytorch=1.12.1<br>
● torchvision=0.13.1<br>
● timm=0.6.11<br>
● albumentations=1.3.0<br>
● imagecorruptions=1.1.2<br>
● h5py=3.7.0<br>
● cython=0.29.32<br>
● yacs=0.1.8<br>

#### Installation
```python
pip install -r requirements.txt
```
If you find some packages are missing, please install them manually.

### 2.Prepare dataset
```python
mkdir data
```
Please download the dataset, and then rename and unzip them under the data<br>
Download the datasets, [Market-1501](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html), [CUHK03](https://openaccess.thecvf.com/content_cvpr_2014/html/Li_DeepReID_Deep_Filter_2014_CVPR_paper.html), [MSMT17](https://arxiv.org/abs/1711.08565).

### 3.Train
Train a CAFT model on Market-1501,
```python
python train.py
```

### 4.Test
Test a CAFT model on Market-1501,
```python
python test.py
```

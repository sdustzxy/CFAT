# Corruption-invariant Person Re-identification Via Coarse-to-Fine Feature Alignment

## Introduction
This paper proposes a coarseto-fine semantic alignment framework that learns corruptioninvariant pedestrian features for re-identification from the perspective of multi-modal feature alignment. In this framework, a Coarse-to-Fine Feature Alignment Transformer (CFAT) is introduced to extract and align features of pedestrian images with different corruptions. The experimental results demonstrate our CFAT outputs state-of-the-arts with a large margin in various corrupted scenes.

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

### 3.Usage
Trained model are provided. You may download it from [Google Drive](https://1drv.ms/f/c/25ca6820bee662c1/EiMEWZ9gU-VLrHCZ18ZfqyoB3AMeupI2NblHcSgeWgk2jQ?e=nDUa6a).

#### 1.Train
Train a CAFT model on Market-1501,
```python
python train.py
```
Parameter settings: .\config\defaults.py

#### 2.Test
Test a CAFT model on Market-1501,
```python
python test.py
```

### 4.Corruption
The visualization results of 20 corruptions.<br>
![5c2588655c30f5b8f126c574a3ccddc1](https://github.com/user-attachments/assets/5d793efd-7e26-45b6-898f-c6f7dae97132)
Each corruption has 5 different severity levels.<br>
<gaussian_noise src="https://github.com/Chen-Yi-Ran/OnlineStudy/blob/master/file/course.jpg" width="210px"><br>
![gaussian_noise jpg](https://github.com/user-attachments/assets/f722e1a0-079b-44c7-92d5-4bec889abdd1)<br>
![brightness](https://github.com/user-attachments/assets/5fe32dd2-831f-4d95-ad19-bb4fa950a97a)

## Citation
```python
@article{10703072,
  author={Zhang, Xinyu and Zhang, Peng and Shan, Caifeng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Corruption-invariant Person Re-identification Via Coarse-to-Fine Feature Alignment}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2024.3472122}}
```

## Acknowledgement
Our code is extended from the following repositories. We thank the authors for releasing the codes.<br>
[TransReID](https://github.com/damo-cv/TransReID)<br>
[CIL-ReID](https://github.com/MinghuiChen43/CIL-ReID)


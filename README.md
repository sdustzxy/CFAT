# Corruption-invariant Person Re-identification Via Coarse-to-Fine Feature Alignment

## Introduction
Corruption-invariant Person Re-identification (CIReID) aims to build robust identity correspondence across nonoverlapped cameras even when severe image corruptions occur. It is challenging as those corruptions contaminate intrinsic pedestrian characteristics and cause semantic misalignment in feature space. To address this issue, this paper proposes a coarseto-fine semantic alignment framework that learns corruptioninvariant pedestrian features for re-identification from the perspective of multi-modal feature alignment. In this framework, a Coarse-to-Fine Feature Alignment Transformer (CFAT) is introduced to extract and align features of pedestrian images with different corruptions. Specifically, the CFAT aligns features of corrupted samples to that of the corresponding clean samples in a knowledge distillation manner in the coarse alignment stage, i.e., a teacher network distils identity-related semantics from clean samples and supervises the student network learning semantic-consistent features from corrupted samples. To avoid information loss of the strict alignment, we propose to integrate a Bridge Feature Generation (BFG) module into CFAT to construct meaningful latent structures among modalities in the fine alignment stage. This enables seamless alignment of the same identity between corrupted and clean modalities, leading to better re-identification performance. To evaluate the effectiveness of the proposed method, extensive experiments are conducted on three public benchmark datasets, i.e., Market-1501, CUHK-03, and MSMT-17. The experimental results demonstrate our CFAT outputs state-of-the-arts with a large margin in various corrupted scenes.

### Method overview
![image](https://github.com/user-attachments/assets/27612791-d121-41ad-aab3-b3b8e0965ecc)

## Quick Start
### 1.Install dependencies
● python=3.8.13
● pytorch=1.12.1
● torchvision=0.13.1
● timm=0.6.11
● albumentations=1.3.0
● imagecorruptions=1.1.2
● h5py=3.7.0
● cython=0.29.32
● yacs=0.1.8

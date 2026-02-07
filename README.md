<div align="center">

<h1> ECGFlowCMR: Pretraining with ECG-Generated Cine CMR Improves Cardiac Disease Classification and Phenotype Prediction </h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.

<h5 align="center"> 

<!-- <a href='https://www.lanxplanet.com/GEM-ECG/'><img src='https://img.shields.io/badge/Homepage-8A2BE2'></a> -->
<a href='https://arxiv.org/pdf/2601.20904'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://huggingface.co/XiaochengFang/ECGFlowCMR/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'>
<a href='https://www.ukbiobank.ac.uk/'><img src='https://img.shields.io/badge/Dataset-Huggingface-yellow'>

<p align="center">
  Xiaocheng Fang, Zhengyao Ding, Guangkun Nie, Jieyi Cai, Yujie Xiao, Bo Liu<br/>
  Jiarui Jin, Haoyu Wang, Shun Huang, Ting Chen, Hongyan Li, Shenda Hong
</p>


<p align="center">
    <img src="pics/intro.png" width="90%">
</p>

</h5>
</div>

## Introduction: 
Cardiac Magnetic Resonance (CMR) imaging provides a comprehensive assessment of cardiac structure and function but remains constrained by high acquisition costs and reliance on expert annotations, limiting the availability of large-scale labeled datasets. In contrast, electrocardiograms (ECGs) are inexpensive, widely accessible, and offer a promising modality for conditioning the generative synthesis of cine CMR. To this end, we propose ECGFlowCMR, a novel ECG-to-CMR generative framework that integrates a Phase-Aware Masked Autoencoder (PA-MAE) and an Anatomy-Motion Disentangled Flow (AMDF) to address two fundamental challenges: (1) the cross-modal temporal mismatch between multi-beat ECG recordings and single-cycle CMR sequences, and (2) the anatomical observability gap due to the limited structural information inherent in ECGs. Extensive experiments on the UK Biobank and a proprietary clinical dataset demonstrate that ECGFlowCMR can generate realistic cine CMR sequences from ECG inputs, enabling scalable pretraining and improving performance on downstream cardiac disease classification and phenotype prediction tasks.

## 🎬 Demo
<img src="videos/50.gif"/>
<img src="videos/388.gif"/>
<img src="videos/484.gif"/>
<img src="videos/1470.gif"/>
<img src="videos/3716.gif"/>
<img src="videos/4424.gif"/>

## ⚙️ ECGFlowCMR Training

<p align="center">
    <img src="pics/method.png" width="90%">
</p>

1. Build environment from requirements.txt
2. Train the appropriate weights using your own dataset.

### Pretraining for Phase-Aware Masked Autoencoder:
```sh
python train_ecg.py
```
### Pretraining for 3D-VAE:
```sh
python train_disentangle_vae3d.py
```
### Training for Anatomy-Motion Disentangled Flow:
```sh
python train_flow.py
```
### Evaluation:
```sh
python evaluation.py
```

## 🚀 Quick Generation
You can use our pre-trained model weights for fast ECG to CMR.

### Generation:
```sh
python infer_flow.py
```

## 📝 Citation:
If you find our work interesting and helpful, please consider giving our repo a star. Additionally, if you would like to cite our work, please use the following format:
```sh
@article{fang2026ecgflowcmr,
  title={ECGFlowCMR: Pretraining with ECG-Generated Cine CMR Improves Cardiac Disease Classification and Phenotype Prediction},
  author={Fang, Xiaocheng and Ding, Zhengyao and Cai, Jieyi and Xiao, Yujie and Liu, Bo and Jin, Jiarui and Wang, Haoyu and Nie, Guangkun and Huang, Shun and Chen, Ting and others},
  journal={arXiv preprint arXiv:2601.20904},
  year={2026}
}
```

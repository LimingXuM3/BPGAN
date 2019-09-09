# Overview

A test implementation for the submitted paper "BPGAN: Bidirectional CT-to-MRI Prediction using Multi-Generative Multi-Adversarial Nets with Spectral Normalization and Localization" (Under reviewing)

# Environment: 
  python 3.6

# Supported Toolkits
  pytorch (Pytorch http://pytorch.org/)
  
  torchvision
  
  numpy
  
  time
  
  pandas
  
  scipy
  
# Demo

  1. Download pre-trained models from [BaiduNetdisk](https://pan.baidu.com/s/1XLjCZnXlRDvmIaHbHAQblA). password: ciw9.

  2. Download partial test samples from [BaiduNetdisk](https://pan.baidu.com/s/1XLjCZnXlRDvmIaHbHAQblA), then put all this data into corresponding dir and extract compressed files.
       
  3. Copy the model (net_G_A: CT predictor, net_G_B: MRI predictor) into your dir
  
     cp latest_net_G_A.pth ./brain_model/  
     
     cp latest_net_G_B.pth ./brain_model/  

  4. Test for CT or MRI precition from MRI or CT images in proposed BPGAN 
     python test.py --dataroot ./datasets/brain --name brain_model

# Notes
- The implementation of proposed BPGAN model is based on cycle-GAN (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We improve the cycle-GAN by introducing pathological auxiliary information, spectral normalization, localization and edge retention to achieve the bidirectional prediction between CT and MRI images.
- This is developed on a Linux machine running Ubuntu 16.04.
- Use GPU for the high speed computation.
- Due to partial samples in SPLP dataset related to private information, so please e-mail me (xulimmail@gmail.com) if you need the dataset and I will share a private link with you.

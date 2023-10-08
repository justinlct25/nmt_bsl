# Neural Machine Translation for British Sign Language

This repository consists of codes for data pre-processing and rearranging on BSL datasets BSLCP and BOBSL. 

Also, code for extracting features from sign language video frames using pre-trained weights of I3D CNN models is include. 

## Pre-processing running files
- bobsl_preprocess:
    - dataset_rearrange.py
- bslcp_preprocess:
    - dataset_rearrange_xxx_to_xxx.py

## Feature extraction running files
- i3d_model_wlasl
    - extraction_i3d_model.py

## Reproduced SLT model training papers
- https://github.com/neccam/slt
- https://github.com/avoskou/Stochastic-Transformer-Networks-with-Linear-Competing-Units-Application-to-end-to-end-SL-Translatio
- https://github.com/YinAoXiong/GASLT
- https://github.com/imatge-upc/slt_how2sign_wicv2023

## BSL datasets
- https://www.robots.ox.ac.uk/~vgg/data/bobsl/
- https://bslcorpusproject.org/cava/

## Feature extraction framework
- https://github.com/dxli94/WLASL


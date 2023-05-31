# PatchNR: Learning from Small Data by Patch Normalizing Flow Regularization

This code belongs to the paper [1] available at https://doi.org/10.1088/1361-6420/acce5e (open access). Please cite the paper, if you use this code.

The repository contains an implementation of patchNRs as introduced in [1]. It contains scripts for reproducing the numerical example CT imaging in Section 4.1 and Superresolution in Section 4.2.

The folder `input_imgs` contains the images for learn the patchNR, a validation image and the test image illustrated in the paper.

For questions and bug reports, please contact Fabian Altekrüger (fabian.altekrueger@hu-berlin.de), Alexander Denker (adenker@uni-bremen.de) or Paul Hagemann (hagemann@math.tu-berlin.de).

## CONTENTS

1. REQUIREMENTS  
2. USAGE AND EXAMPLES
3. REFERENCES

## 1. REQUIREMENTS

The code requires several Python packages. We tested the code with Python 3.9.7 and the following package versions:

- pytorch 1.10.0
- numpy 1.21.2
- tqdm 4.62.3
- scipy 1.7.1
- freia 0.2
- dival 0.6.1
- odl 1.0.0

Usually the code is also compatible with some other versions of the corresponding Python packages.

## 2. USAGE AND EXAMPLES

You can start the training of the patchNR by calling the script `train_patchNR.py`. There you can choose between the different image classes. 

### CT IMAGING

The script `patchNR_CT.py` is the implementation of the CT example in [1, Section 4.1]. Here you can choose between the full angle and the limited angle case. The used data is from the LoDoPaB dataset [2], which is available at https://zenodo.org/record/3384092##.Ylglz3VBwgM.

### SUPERRESOLUTION

The script `patchNR_superres.py` is the implementation of the superresolution example in [1, Section 4.2]. The used images of material microstructures have been acquired in the frame of the EU Horizon 2020 Marie Sklodowska-Curie Actions Innovative Training Network MUMMERING (MUltiscale, Multimodal and Multidimensional imaging for EngineeRING, Grant Number 765604) at the beamline TOMCAT of the SLS by A. Saadaldin, D. Bernard, and F. Marone Welford. The low-resolution image used for reconstruction is generated by artificially downsampling and adding Gaussian noise. For more details on the downsampling process, see [1, Section 4.2]. 

### ZERO-SHOT SUPERRESOLUTION

The scripts `patchNR_zeroshot.py` and `patchNR_zeroshot_material.py` are the implementations of the zero-shot superresolution example in [1, Section 4.2]. The patchNR was tested on the BSD68 dataset [3] as well as on material microstructures and the low-resolution image used for reconstruction is generated by artificially downsampling and adding Gaussian noise. For more details on the downsampling process, see [1, Section 4.2]. 

## 3. REFERENCES

[1] F. Altekrüger, A. Denker, P. Hagemann, J. Hertrich, P. Maass and G. Steidl.  
PatchNR: Learning from Very Few Images by Patch Normalizing Flow Regularization.   
Inverse Problems, vol. 39, no. 6, 2023

[2] J. Leuschner, M. Schmidt, D. O. Baguer and P. Maass.  
LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction.  
Scientific Data, 9(109), 2021.

[3] D. Martin, C. Fowlkes, D. Tal, and J. Malik.  
A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics.  
Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, volume 2, pages 416–423. IEEE, 2001.

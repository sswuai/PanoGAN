[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/BiGraphGAN/blob/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6.9-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-1.0.0-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Ha0Tang/BiGraphGAN)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Ha0Tang/BiGraphGAN/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

## Contents
  - [PanoGAN](#PanoGAN)
  - [Installation](#Installation)
  - [Dataset Preparation](#Dataset-Preparation)
  - [Generating Images Using Pretrained Model](#Generating-Images-Using-Pretrained-Model)
  - [Train and Test New Models](#Train-and-Test-New-Models)
  - [Download Images Produced by the Authors](#Download-Images-Produced-by-the-Authors)
  - [Evaluation](#Evaluation)
  - [Acknowledgments](#Acknowledgments)
  - [Related Projects](#Related-Projects)
  - [Citation](#Citation)
  - [Contributions](#Contributions)

## PanoGAN

**[Panorama Generative Adversarial Network for Cross-View Panorama Image Synthesis](https://arxiv.org/abs/2008.04381)**  
[Songsong Wu](https://www.researchgate.net/profile/Songsong_Wu)<sup>1</sup>, [Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>2</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>24</sup>. <br> 
<sup>1</sup> Guangdong University of Petrochemical Technology, China, <sup>2</sup>University of Trento, Italy.<br>
Submited to [IEEE Transactions on MultiMedia]. <br>

### Framework
<img src='./imgs/panogan_architecture_0001250.jpg' width=800>

### Generator
<img src='./imgs/G_D_v1.jpg' width=800>

### Discriminator
<img src='./imgs/g_d.jpg' width=800>

### Comparison Results on CVUSA Dataset

<img src='./imgs/cvusa_sota_comparison.jpg' width=800>

### Comparison Results on CVUSA Dataset

<img src='./imgs/op_sota_comparison.jpg' width=800>

### [License](./LICENSE.md)

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [sswuai@gmail.com](sswuai@gmail.com).

## Installation

Clone this repo.
```bash
git clone https://github.com/sswuai/PanoGAN.git
cd PanoGAN/
```

This code requires PyTorch 1.0.0 and python 3.6.9+. Please install the following dependencies:
* pytorch 1.0.0
* torchvision
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm
* dominate

To reproduce the results reported in the paper, you need to run experiments with at least 1 NVIDIA 11GB 2080Ti GPU.

## Dataset Preparation

Please follow [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN/tree/master/person_transfer#data-preperation) to directly download both Market-1501 and DeepFashion datasets.

## Generating Images Using Pretrained Model
### PanoGAN trained with 35,548 aerial-panorama image pairs on CVUSA dataset
```bash
cd scripts/
sh download_bigraphgan_model.sh market
cd ..
cd market_1501/
```
Then,
1. Change several parameters in `test_market_pretrained.sh`.
2. Run `sh test_market_pretrained.sh` for testing.

### PanoGAN tranined on OP dataset
```bash
cd scripts/
sh download_bigraphgan_model.sh deepfashion
cd ..
cd deepfashion/
```
Then,
1. Change several parameters in `test_deepfashion_pretrained.sh`.
2. Run `sh test_deepfashion_pretrained.sh` for testing.

## Train and Test New Models
### CVUSA dataset
1. Go to the [scripts](https://github.com/sswuai/PanoGAN/tree/master/scripts) folder. 
2. Change several parameters in `train_market.sh`.
3. Run `sh train_market.sh` for training.
4. Change several parameters in `test_market.sh`.
5. Run `sh test_market.sh` for testing.

### OP dataset
1. Go to the [scripts](https://github.com/sswuai/PanoGAN/tree/master/scripts) folder. 
2. Change several parameters in `train_deepfashion.sh`.
3. Run `sh train_deepfashion.sh` for training.
4. Change several parameters in `test_deepfashion.sh`.
5. Run `sh test_deepfashion.sh` for testing.

## Download Images Produced by the Authors
**For your convenience, you can directly download the images produced by the authors for qualitative comparisons in your own papers!!!**

### CVUSA dataset
```bash
cd scripts/
sh download_bigraphgan_result.sh market
```

### OP dataset
```bash
cd scripts/
sh download_bigraphgan_result.sh deepfashion
```

## Evaluation
We adopt Prediction Accuracy, Inception Score, KL Score, SSIM, PSNR, and SD for evaluation of all the involved methods. Please refer to [Evaluation](https://github.com/tengteng95/Pose-Transfer#evaluation) for more details.
 
## Acknowledgments
This source code is inspired by both [Pix2Pix](https://github.com/phillipi/pix2pix.git) and [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN). 

## Related Projects
**[SelectionGAN](https://github.com/Ha0Tang/SelectionGAN) | [Guided-I2I-Translation-Papers](https://github.com/Ha0Tang/Guided-I2I-Translation-Papers)**

## Citation
If you use this code for your research, please cite our papers.

```
@inproceedings{tang2019multi,
  title={Multi-channel attention selection gan with cascaded semantic guidance for cross-view image translation},
  author={Tang, Hao and Xu, Dan and Sebe, Nicu and Wang, Yanzhi and Corso, Jason J and Yan, Yan},
  booktitle={CVPR},
  year={2019}
}

```

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Songsong Wu (sswuai@gmail.com](sswuai@gmail.com)).

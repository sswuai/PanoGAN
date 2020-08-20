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
[Songsong Wu](https://www.researchgate.net/profile/Songsong_Wu)<sup>1</sup>, [Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>23</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>24</sup>. <br> 
<sup>1</sup> Guangdong University of Petrochemical Technology, China, <sup>2</sup>University of Trento, Italy, <sup>3</sup>University of Oxford, UK, <sup>4</sup>Huawei Research Ireland, Ireland.<br>
Submited to [IEEE Transactions on MultiMedia]. <br>

### Framework
<img src='./imgs/panogan_architecture_0001250.jpg' width=1200>

### Framework
<img src='./imgs/method.jpg' width=1200>

### Comparison Results

<img src='./imgs/market_results.jpg' width=1200>

<br>

<img src='./imgs/fashion_results.jpg' width=1200>

### [License](./LICENSE.md)

Copyright (C) 2019 University of Trento, Italy.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [hao.tang@unitn.it](hao.tang@unitn.it).

## Installation

Clone this repo.
```bash
git clone https://github.com/Ha0Tang/BiGraphGAN
cd BiGraphGAN/
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

To reproduce the results reported in the paper, you need to run experiments on NVIDIA DGX1 with 4 32GB V100 GPUs for DeepFashion, and 1 32GB V100 GPU for Market-1501.

## Dataset Preparation

Please follow [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN/tree/master/person_transfer#data-preperation) to directly download both Market-1501 and DeepFashion datasets.

## Generating Images Using Pretrained Model
### Market-1501
```bash
cd scripts/
sh download_bigraphgan_model.sh market
cd ..
cd market_1501/
```
Then,
1. Change several parameters in `test_market_pretrained.sh`.
2. Run `sh test_market_pretrained.sh` for testing.

### DeepFashion
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
### Market-1501
1. Go to the [market_1501](https://github.com/Ha0Tang/BiGraphGAN/tree/master/market_1501) folder. 
2. Change several parameters in `train_market.sh`.
3. Run `sh train_market.sh` for training.
4. Change several parameters in `test_market.sh`.
5. Run `sh test_market.sh` for testing.

### DeepFashion
1. Go to the [deepfashion](https://github.com/Ha0Tang/BiGraphGAN/tree/master/deepfashion) folder. 
2. Change several parameters in `train_deepfashion.sh`.
3. Run `sh train_deepfashion.sh` for training.
4. Change several parameters in `test_deepfashion.sh`.
5. Run `sh test_deepfashion.sh` for testing.

## Download Images Produced by the Authors
**For your convenience, you can directly download the images produced by the authors for qualitative comparisons in your own papers!!!**

### Market-1501
```bash
cd scripts/
sh download_bigraphgan_result.sh market
```

### DeepFashion
```bash
cd scripts/
sh download_bigraphgan_result.sh deepfashion
```

## Evaluation
We adopt SSIM, mask-SSIM, IS, mask-IS, and PCKh for evaluation of Market-1501. SSIM, IS, PCKh for DeepFashion. Please refer to [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer#evaluation) for more details.
 
## Acknowledgments
This source code is inspired by both [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer), [GloRe](https://github.com/facebookresearch/GloRe) and [SelectionGAN](https://github.com/Ha0Tang/SelectionGAN). 

## Related Projects
**[SelectionGAN](https://github.com/Ha0Tang/SelectionGAN) | [Guided-I2I-Translation-Papers](https://github.com/Ha0Tang/Guided-I2I-Translation-Papers)**

## Citation
If you use this code for your research, please cite our papers.

SelectionGAN
```
@inproceedings{tang2019multi,
  title={Multi-channel attention selection gan with cascaded semantic guidance for cross-view image translation},
  author={Tang, Hao and Xu, Dan and Sebe, Nicu and Wang, Yanzhi and Corso, Jason J and Yan, Yan},
  booktitle={CVPR},
  year={2019}
}

@article{tang2020multi,
  title={Multi-channel attention selection gans for guided image-to-image translation},
  author={Tang, Hao and Xu, Dan and Yan, Yan and Corso, Jason J and Torr, Philip HS and Sebe, Nicu},
  journal={arXiv preprint arXiv:2002.01048},
  year={2020}
}
```

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Hao Tang ([hao.tang@unitn.it](hao.tang@unitn.it)).

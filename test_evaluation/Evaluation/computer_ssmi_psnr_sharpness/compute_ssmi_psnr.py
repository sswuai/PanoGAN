# code derived from PlacesCNN for scene classification Bolei Zhou

import pytorch_ssim
import torch
import os
import cv2
import numpy as np
import sys
import math

def psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return 10 * math.log10(255.0 * 255.0 / mse)

# load the test image
img_real = sys.argv[1].strip()
img_synthesized = sys.argv[2].strip()
print (img_real)
print (img_synthesized)

path = list(os.walk(img_real))
path_t = path[0][0]
root_t = path[0][1]
files_t = path[0][2]

n = len(files_t)
print(n)
sum_ssim=0
sum_psnr=0

for i in range(n):
    img_name = files_t[i]
    img_name = os.path.splitext(img_name)[0]

    img_fake_name = img_name[0:7]#+'_fake_B'
    img_path_real = img_real + '/' + img_name + '.jpg'
    img_path_synthesized = img_synthesized + '/' + img_fake_name + '.png'

    #print(img_path_real)
    #print(img_path_synthesized)
    npImg1 = cv2.imread(img_path_real)
    npImg2 = cv2.imread(img_path_synthesized)
    sum_psnr = psnr(npImg1,npImg2)+sum_psnr


    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0) / 255.0
    img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0) / 255.0

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()

        sum_ssim = sum_ssim+pytorch_ssim.ssim(img1, img2).item()


    if(i%500==0):
        print(i)

ss = sum_ssim/n
ps = sum_psnr/n

print("SSMI:", ss)
print("PSNR:", ps)


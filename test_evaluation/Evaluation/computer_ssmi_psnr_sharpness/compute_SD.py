# code derived from PlacesCNN for scene classification Bolei Zhou


import os
import cv2
import numpy as np
import sys



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
SD_SUM=0
for i in range(n):
    img_name = files_t[i]
    img_name = os.path.splitext(img_name)[0]

    img_fake_name = img_name[0:7]+'_fake_B'
    img_path_real = img_real + '/' + img_name + '.png'
    img_path_synthesized = img_synthesized + '/' + img_fake_name + '.png'

    npImg1 = cv2.imread(img_path_real)
    npImg2 = cv2.imread(img_path_synthesized)

    img1_HLS = cv2.cvtColor(npImg1, cv2.COLOR_BGR2HLS)
    img2_HLS = cv2.cvtColor(npImg2, cv2.COLOR_BGR2HLS)
    L1 = img1_HLS[:, :, 1]
    L2 = img2_HLS[:, :, 1]

    u1 = np.mean(L1)
    u2 = np.mean(L2)
    LP1 = cv2.Laplacian(L1, cv2.CV_64F).var()
    LP2 = cv2.Laplacian(L2, cv2.CV_64F).var()
    s1 = np.sum(LP1 / u1)
    s2 = np.sum(LP2 / u2)
    sd = abs(s1-s2)
    SD_SUM = SD_SUM+sd

    if(i%500==0):
        print(i)
SD = SD_SUM/n
print(SD)




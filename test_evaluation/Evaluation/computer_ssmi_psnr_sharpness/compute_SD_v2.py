import cv2
import numpy as np
import math
import os
import sys

def brenner(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out+=(int(img[x+2,y])-int(img[x,y]))**2
    return out


def Laplacian(img):
    return cv2.Laplacian(img,cv2.CV_64F).var()


def SMD(img):
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            out+=math.fabs(int(img[x,y])-int(img[x,y-1]))
            out+=math.fabs(int(img[x,y]-int(img[x+1,y])))
    return out

def SMD2(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

def variance(img):
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

def energy(img):
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)*((int(img[x,y+1]-int(img[x,y])))**2)
    return out

def Vollath(img):
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

def entropy(img):
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out


img_real = sys.argv[1].strip()
img_synthesized = sys.argv[2].strip()
print(img_real)
print(img_synthesized)

path = list(os.walk(img_real))
path_t = path[0][0]
root_t = path[0][1]
files_t = path[0][2]

n = len(files_t)
print(n)
sum_br=0
sum_la=0
sum_sm=0
sum_sm2=0
sum_va=0
sum_en=0
sum_vo=0
sum_ent=0


for i in range(n):
    img_name = files_t[i]
    img_name = os.path.splitext(img_name)[0]

    img_fake_name = img_name[0:7]+'_fake_B'
    img_path_real = img_real + '/' + img_name + '.png'
    img_path_synthesized = img_synthesized + '/' + img_fake_name + '.png'

    npImg1 = cv2.imread(img_path_real)
    npImg2 = cv2.imread(img_path_synthesized)

    img1 = cv2.cvtColor(npImg1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(npImg2, cv2.COLOR_BGR2GRAY)

    sum_br=sum_br+abs(brenner(img1)-brenner(img2))
    sum_la=sum_la+abs(Laplacian(img1)-Laplacian(img2))
    sum_sm=sum_sm+abs(SMD(img1)-SMD(img2))
    sum_sm2=sum_sm2+abs(SMD2(img1)-SMD2(img2))
    sum_va=sum_va+abs(variance(img1)-variance(img2))
    sum_en=sum_en+abs(energy(img1)-energy(img2))
    sum_vo=sum_vo+abs(Vollath(img1)-Vollath(img2))
    sum_ent=sum_ent+abs(entropy(img1)-entropy(img2))

    if (i % 50 == 0):
        print(i)


br=sum_br/n
la=sum_la/n
sm=sum_sm/n
sm2=sum_sm2/n
va=sum_va/n
en=sum_en/n
vo=sum_vo/n
ent=sum_ent/n

print('Brenner:', br)
print('Laplacian:', la)
print('SMD:', sm)
print('SMD2:', sm2)
print('Variance:', va)
print('Energy:', en)
print('Vollath:', vo)
print('Entropy:',ent)

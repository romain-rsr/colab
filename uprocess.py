import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import h5py
import scipy.io as sio

def square_centered_crop(img):
    w,h=img.size
    size=min(w,h)-1

    x1=int(w/2-size/2)
    y1=int(h/2-size/2)
    x2=int(w/2+size/2)
    y2=int(h/2+size/2)

    #print("\t u",img.size,"-",w,h,"-",x1,y1,x2,y2)
    if x1>=0 and y1>=0 and x2<w and y2<h:
        img=img.crop((x1, y1, x2, y2))
        bvalid=1
    else:
        bvalid=0
    if bvalid==0:print("error",x1,y1,x2,y2,"-",w,h)
    #print("\t u",img.size)

    return img,bvalid

# n output files
nfiles=50000

# pdata
if 1:pdata="data"

# vsize
target_size = 128

# load + resize images
images=[]
sfiles=os.listdir(pdata)
k=0
for sfile in sfiles[:min(len(sfiles),nfiles)]:
    if k%1000==0:print(k)
    k+=1
    img=Image.open(os.path.join(pdata, sfile)) # open image
    img,bvalid=square_centered_crop(img)
    img=img.resize((target_size, target_size), resample=Image.BILINEAR) # resize image    
    img=np.asarray(img) # format image 
    if len(np.shape(img))==3:
        img=img.transpose((2, 0, 1)) # transpose image
        img=img.reshape((1, 3, target_size, target_size)) # reshape image
        images.append(img)

images=np.concatenate(images)

# create h5 file
file = h5py.File('data/celeba_wild/celeba_wild.h5', "w")
file.create_dataset("train_img", np.shape(images), h5py.h5t.STD_U8BE, data=images)
file.close()


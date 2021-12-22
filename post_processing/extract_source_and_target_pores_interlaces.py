# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:11:35 2021

@author: firo
"""

import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt


baseFolder = r"A:\Robert_TOMCAT_4"


def load_slice(sample, z=0, baseFolder = baseFolder):
    sourceFolder = os.path.join(baseFolder, sample, '05b_labels_split_v2')
    imlist = os.listdir(sourceFolder)
    if 'Thumbs.db' in imlist: imlist.remove('Thumbs.db')
    imname = imlist[z]
    print(imname)
    path = os.path.join(sourceFolder, imname)
    img = io.imread(path)
    return img


def get_leg_vals(im,loc, vert=True):
    shp = im.shape
    if vert:
        im1 = im[:,:int(shp[1]/2)]
        im2 = im[:,int(shp[1]/2):]
    else:
        im1 = im[:int(shp[0]/2),:]
        im2 = im[int(shp[0]/2):,:]
    plt.figure()
    plt.imshow(im1)
    plt.title(loc+'im1')
    plt.figure()
    plt.imshow(im2)
    plt.title(loc+'im2')
    v1 = np.unique(im1)[1:]
    v2 = np.unique(im2)[1:]
    return v1,v2

def leg_unique_values(sample, baseFolder=baseFolder, z1=0, z2=-1, bot_vert=False, top_vert=True):
    print(sample)
    top_im = load_slice(sample, z=z1, baseFolder=baseFolder)
    t1, t2 = get_leg_vals(top_im, 'top', vert=top_vert)
    bot_im = load_slice(sample, z=z2, baseFolder=baseFolder)
    s1, s2 = get_leg_vals(bot_im, 'bot', vert=bot_vert)
    return s1, s2, t1, t2
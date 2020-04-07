# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:02:16 2019

Create pore space to extract pores


@author: firo
"""

import sys
library=r"R:\Scratch\305\_Robert\Python_Library"

if library not in sys.path:
    sys.path.append(library)



# FIXME: use PMM to create pore space of T4-samples

import os
import numpy as np
# from scipy.ndimage import morphology
import imageio
#import skimage
from skimage import morphology as skmorph
from joblib import Parallel, delayed
import multiprocessing as mp
import robpylib

#baseFolder = 'X:\TOMCAT3_processing_1'
#newBaseFolder = 'U:\TOMCAT_3_segmentation'
#baseFolder = r'Z:\Robert_TOMCAT_3_Part_2'
baseFolder = r'Z:\Robert_TOMCAT_3'
newBaseFolder = baseFolder


excluded_samples=[
 'T3_100_4',        #FOV moved during acquisition
 'T3_300_6',        # water only in little pores
 'T3_300_3_III',    #no water
 'T4_025_3_III',    #wet from start
 'T3_300_8',        #FOV moved
 'T3_025_7',        #water only in small pores
 'T3_025_4_III',    #little water, data set incomplete, missing time steps reconstructed at PSI-ra-cluster, but only little water -> not reasonable to process?!
 'T3_025_3_II'      #little, water wet with acetone
 ]

#excluded_samples=[]

def yarn_pores(fiberFolder, targetfolder, name):
    fiber = imageio.imread(os.path.join(fiberFolder,name))
    fiber = fiber>0

    hull = skmorph.convex_hull_image(fiber)
    pores = np.bitwise_xor(hull, fiber)
    pores = skmorph.remove_small_objects(pores, min_size=5, connectivity=1)
    pores = (pores*255).astype(np.uint8)
    imageio.imsave(os.path.join(targetFolder,name), pores)

def interlace_masking(Stack, maskingthreshold=11000):
    ref_mean = Stack[:,:,:1000].mean()
    for z in range(Stack.shape[2]):
        Stack[:,:,z] = Stack[:,:,z] - (Stack[:,:,z].mean() - ref_mean)
    Stack[np.where(Stack<0)] = 0
    Stack[Stack<maskingthreshold]=0
    Stack[Stack>0]=1
    return Stack   

def interlace_pores(fiberFolder, watermask, targetfolder, name):
    fiber = imageio.imread(os.path.join(fiberFolder,name))
    fiber = fiber>0

    
    water = watermask
    
    pores = np.bitwise_xor(water,fiber)
    pores = skmorph.remove_small_objects(pores, min_size=5, connectivity=1)
    pores = skmorph.binary_erosion(pores)
    pores = (pores*255).astype(np.uint8)
    imageio.imsave(os.path.join(targetFolder,name), pores)


c=0

for sample in os.listdir(baseFolder):
#    if not sample == 'T4_300_5_III': continue
    # if not sample[1] == '4': continue
#    if not sample == 'T3_025_9_III': continue
    if not sample in robpylib.TOMCAT.INFO.samples_to_repeat: continue
    if sample == 'T4_025_2_II': continue
    if sample in excluded_samples:
        c=c+1
        continue
    print(sample)
    fiberFolder=os.path.join(baseFolder,sample,'01a_weka_segmented_dry','classified')
    sourceFolder=os.path.join(baseFolder,sample,'02_pystack_registered_from_5')
    # waterFolder=os.path.join(baseFolder,sample,'03_gradient_filtered_transitions') #for the T4 samples, use final water configuration
    waterFolder=os.path.join(sourceFolder,os.listdir(sourceFolder)[-1])
    
    if not newBaseFolder:
        newBaseFolder=baseFolder
    
    targetFolder=os.path.join(newBaseFolder,sample,'04a_void_space_from_5')
    
#    if os.path.exists(targetFolder):
#        c=c+1
#        continue
    
    if not os.path.exists(os.path.join(newBaseFolder,sample)):
        os.mkdir(os.path.join(newBaseFolder,sample))
    
    if not os.path.exists(targetFolder):
        os.mkdir(targetFolder)
    
    fibernames = os.listdir(fiberFolder)
    if 'Thumbs.db' in fibernames: fibernames.remove('Thumbs.db')
    
    num_cores=mp.cpu_count()
    
    if sample[1]=='3':
        Parallel(n_jobs=num_cores)(delayed(yarn_pores)(fiberFolder, targetFolder, name) for name in fibernames)

    if sample[1]=='4':
        masks, names  = robpylib.CommonFunctions.ImportExport.ReadStackNew(waterFolder)
        masks = (masks > 0).astype(np.uint8)
        # Parallel(n_jobs=num_cores)(delayed(interlace_pores)(fiberFolder, masks[:,:,z], targetFolder, fibernames[z]) for z in range(len(fibernames)))
        robpylib.CommonFunctions.ImportExport.WriteStackNew(targetFolder, names, masks)

    
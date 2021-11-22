# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:09:41 2021

@author: firo
"""

import os
from scipy import ndimage
import numpy as np
import pandas as pd
import robpylib
from joblib import Parallel, delayed
# import xarray as xr
# from skimage.morphology import square, disk
# import cupy as cp
# from cupyx.scipy import ndimage as ndi
# from cucim.skimage.morphology import ball as gpuball
from skimage.morphology import ball


drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_4')
destination = os.path.join(drive, 'Robert_TOMCAT_4_netcdf4_split_v2')

baseFolder = r"A:\Robert_TOMCAT_4"
destination = r"A:\Robert_TOMCAT_4_netcdf4_split_v2"
overWrite = False
temp_folder= r"Z:\users\firo\joblib_tmp"


knots={}
knots['T4_025_3_III']=[1244,1572]   #wet from start
knots['T4_100_2_III']=[1204,1450]
knots['T4_300_3_III']=[1431,1664]
knots['T4_025_3']=[1032,1362]
knots['T4_025_4']=[777,1020]
knots['T4_100_3']=[171,405]         #on second small external HDD (rearguard)
knots['T4_100_4']=[233,441]
knots['T4_100_5']=[987,1229]
knots['T4_300_1']=[571,837]         
knots['T4_300_5']=[581,815]
knots['T4_025_1_III']=[1398,1664]
knots['T4_025_2_II']=[149,369]
knots['T4_025_3']=[1026,1341]
knots['T4_025_4']=[794,1017]              #on small external HD (rearguard)
knots['T4_025_5_III']=[1280,1573]    #wet from start    
knots['T4_300_2_II']=[157,460]
knots['T4_300_4_III']=[1272,1524]
knots['T4_300_5_III']=[1377,1610]

def yarn_labeling(im):
    close = ndimage.morphology.binary_closing(im, iterations = 10)
    close = ndimage.morphology.binary_fill_holes(close)
    # dilate = ndimage.morphology.binary_dilation(close, structure = disk(5))
    return close

def track_yarn_affiliation(sample, baseFolder=baseFolder):
    # path = r"A:\Robert_TOMCAT_4\T4_025_1_III\06_fiber_tracing\T4_025_1_III.CorrelationLines.xlsx"
# fiber_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\01a_weka_segmented_dry\classified"
# label_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\05b_labels"
    k1 = knots[sample][0]
    k2 = knots[sample][1]
    
    path = os.path.join(baseFolder, sample, '06_fiber_tracing', ''.join([sample,".CorrelationLines.xlsx"]))
    targetFolder = os.path.join(baseFolder, sample, '06c_yarn_labels')
    fiber_path = os.path.join(targetFolder, 'all_fibers')
    if not os.path.exists(targetFolder):
        os.mkdir(targetFolder)
    segments = pd.read_excel(path, sheet_name = 2)
    # points = pd.read_excel(path, sheet_name = 1)
    nodes = pd.read_excel(path, sheet_name = 0)
    
    top_yarn = []
    bottom_yarn = []
    
    for segment in segments['Segment ID']:
        node_ids = [segments['Node ID #1'][segment], segments['Node ID #2'][segment]]
        position = nodes['Z Coord'][node_ids].mean()
        
        if position > 1012:
            top_yarn.append(segment)
        else:
            bottom_yarn.append(segment)
                
    fibers, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiber_path, track=False)
    shp = fibers.shape

    
    # both_yarn = list(segments['Segment ID'])
    print('asign fibers')
    yarn1 = np.zeros(shp, dtype=np.uint8)
    
    for fiber in top_yarn:
        yarn1[fibers==fiber+1] = 1
        
    
    yarn2 = np.zeros(shp, dtype=np.uint8)
     
    for fiber in bottom_yarn:
         yarn2[fibers==fiber+1] = 1  
         
    targetfiber1 = os.path.join(targetFolder,'yarn1_fibers')
    targetfiber2 = os.path.join(targetFolder,'yarn2_fibers')
    
    robpylib.CommonFunctions.ImportExport.WriteStackNew(targetfiber1, names, yarn1)
    robpylib.CommonFunctions.ImportExport.WriteStackNew(targetfiber2, names, yarn2)
    yarn1 = yarn1>0
    yarn2 = yarn2>0
    
    print('expand fibers of bottom yarn - skipped')
    # print('1 - make one solid piece')
    # results = Parallel(n_jobs = 16, temp_folder = temp_folder)(delayed(yarn_labeling)(yarn2[:,:,z]) for z in range(yarn1.shape[2]))
    # label = np.array(results).transpose(1,2,0)
    # print('2 - 3D dilation')
    # # gpu_array = cp.array(label[:,:,k1-50:].copy())
    # # cut1 = 2016-int(k1/2)
    
    # # gpu_array = cp.array(label[:,:,k1-50:cut1])
    # # cpu_array1 = cp.asnumpy(ndi.morphology.binary_dilation(gpu_array, structure=gpuball(9)))
    
    # # gpu_array = cp.array(label[:,:,cut1:])
    # # cpu_array2 = cp.asnumpy(ndi.morphology.binary_dilation(gpu_array, structure=gpuball(9)))
    
    # # del gpu_array
    
    # # label[:,:,k1-50:cut1] = cpu_array1
    # # label[:,:,cut1:] = cpu_array2
    # # cpu_array = cp.asanumpy(ndi.morphology.binary_dilation(gpu_array, structure=gpuball(8)))
    
    # cpu_array = label
    # cpu_array = ndimage.morphology.binary_dilation(cpu_array, structure=ball(10))
    
    # label = cpu_array
    
    
    # target1 = os.path.join(targetFolder,'yarn1')
    # if not os.path.exists(target1):
    #     os.mkdir(target1)
    # robpylib.CommonFunctions.ImportExport.WriteStackNew(target1, names, label.astype(np.uint8))
    
    # print('expand fibers of top yarn')
    # print('1 - make one solid piece')
    # results = Parallel(n_jobs = 16, temp_folder = temp_folder)(delayed(yarn_labeling)(yarn1[:,:,z]) for z in range(yarn2.shape[2]))
    # label = np.array(results).transpose(1,2,0)
    
    # print('2 - 3D dilation skipped')
    # cut2 = int(k2/2)
    
    # gpu_array = cp.array(label[:,:,:cut2])
    # cpu_array1 = cp.asnumpy(ndi.morphology.binary_dilation(gpu_array, structure=gpuball(9)))
    
    # gpu_array = cp.array(label[:,:,cut2:k2+50])    
    # cpu_array2 = cp.asnumpy(ndi.morphology.binary_dilation(gpu_array, structure=gpuball(9)))
    
    # del gpu_array
    
    # label[:,:,:cut2] = cpu_array1
    # label[:,:,cut2:k2+50] = cpu_array2
    
    # cpu_array = label
    # cpu_array = ndimage.morphology.binary_dilation(cpu_array, structure=ball(10))
    # label = cpu_array
    
    # target2 = os.path.join(targetFolder,'yarn2')
    # if not os.path.exists(target2):
    #     os.mkdir(target2)
    # robpylib.CommonFunctions.ImportExport.WriteStackNew(target2, names, label.astype(np.uint8))
    

samples = os.listdir(baseFolder)

if '.DS_Store' in samples:
    samples.remove('.DS_Store')
# track_yarn_affiliation('T4_025_1_III')
# num_jobs = 2
# results = Parallel(n_jobs=num_jobs, temp_folder=temp_folder)(delayed(track_yarn_affiliation)(sample) for sample in samples)    
samples.reverse()
for sample in samples:
    # if sample == 'T4_025_4': continue
#     if sample == 'T4_100_2_III': continue
#     if sample == 'T4_100_3': continue
#     if sample == 'T4_300_1': continue
#     if sample == 'T4_300_3_III': continue
#     if sample == 'T4_025_2_II': continue
    # if sample == 'T4_025_1_III': continue
#     if sample == 'T4_025_3': continue
#     if sample == 'T4_100_4': continue
    print(sample)
    track_yarn_affiliation(sample)
    
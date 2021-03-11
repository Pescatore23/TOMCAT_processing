# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:38:02 2019

creates substacks of x and y slices at interlace for improved ML segmentation



@author: firo
"""

import os
import numpy as np
import robpylib


def makeindex(indexmax):
    index=[None]*indexmax
    for z in range(indexmax):
        index[z]=z
    index=[''.join([str(item).zfill(4),'.tif']) for item in index]
    return index

baseFolder = r"E:\Robert\Robert_TOMCAT_2"

samples=os.listdir(baseFolder)

excluded_samples=[
 'T3_100_4',        #FOV moved during acquisition
 'T3_300_6',        # water only in little pores
 'T3_300_3_III',    #no water
 'T4_025_3_III',    #wet from start
 'T3_300_8',        #FOV moved
 'T3_025_7',        #water only in small pores
 'T3_025_4_III',    #little water, data set incomplete, missing time steps reconstructed at PSI-ra-cluster, but only little water -> not reasonable to process?!
 'T3_025_3_II',      #little, water wet with acetone
 'T4_025_5_III',    #wet from start
 ]

# sourceFolder = '00_raw'
sourceFolder = '02_pystack_registered'

z_low=1414
z_high=1644

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
knots['R_m8_50_200_1_II'] = [866,1009]



for sample in samples:
    if sample in excluded_samples: continue
    if sample[1]=='3':continue
    z_low, z_high=knots[sample]
    print(sample)
    content=os.listdir(os.path.join(baseFolder, sample, sourceFolder))
    content.sort()
    dryscan=content[0]
    folder=os.path.join(baseFolder, sample, sourceFolder, dryscan)
#    folder = os.path.join(baseFolder, sample, '02a_temporal_mean')
    targetFolder = os.path.join(baseFolder,sample,"01a_weka_segmented_dry") 
#    targetFolder = os.path.join(baseFolder,sample,"01a_weka_segmented_mean")
    
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    
    x_folder=os.path.join(baseFolder, sample, targetFolder, "x_set")
    y_folder=os.path.join(baseFolder, sample, targetFolder, "y_set")
    
    if not os.path.exists(x_folder):
        os.makedirs(x_folder)
    if not os.path.exists(y_folder):
        os.makedirs(y_folder)
    
#    z_low,z_high=interlace_bounds[sample]
    
    substack,names=robpylib.CommonFunctions.ImportExport.ReadStackNew(folder)
    substack=substack[:,:,z_low:z_high]
    names=names[z_low:z_high]
    
    x_stack=np.transpose(substack,(2,1,0))
    y_stack=np.transpose(substack,(0,2,1))
    
    
    x_names=makeindex(x_stack.shape[2])
    y_names=makeindex(y_stack.shape[2])
    
    robpylib.CommonFunctions.ImportExport.WriteStackNew(x_folder, x_names, x_stack)
    robpylib.CommonFunctions.ImportExport.WriteStackNew(y_folder, y_names, y_stack)
    
    
    
    
    
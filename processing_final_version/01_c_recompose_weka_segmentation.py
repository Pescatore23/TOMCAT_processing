# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:18:39 2019

loads probability maps of x-,y-, and z-slices and rebuilds a segmented image


@author: firo
"""

confidence_value=0.55

#confidence_value = 0.7   # for weka on temporal mean (dask enhanced)

import os
import numpy as np
from skimage import io
import robpylib


z_low=1414
z_high=1644

baseFolder = r"E:\Robert_TOMCAT_5_split"
#baseFolder = r'F:\Zwischenlager_Robert\TOMCAT_3'
# baseFolder = 'X:\TOMCAT3_processing_1'
samples=os.listdir(baseFolder)

repeats = robpylib.TOMCAT.INFO.samples_to_repeat


#knots={}
#knots['T4_025_3_III']=[1244,1572]   #wet from start
#knots['T4_100_2_III']=[1204,1450]
#knots['T4_300_3_III']=[1431,1664]
#knots['T4_025_3']=[1032,1362]
#knots['T4_025_4']=[777,1020]
#knots['T4_100_3']=[165,412]         #registration incomplete
#knots['T4_100_4']=[233,441]
#knots['T4_100_5']=[987,1229]
#knots['T4_300_1']=[571,837]         #remove from TOMCAT_3_reconstructions/disk2 (ddm04672)
#knots['T4_300_5']=[581,815]
#knots['T4_025_1_III']=[1398,1664]
#knots['T4_025_2_II']=[149,369]
#knots['T4_025_3']=[1026,1341]
#knots['T4_025_4']=[794,1027]              #reconstruction incomplete
#knots['T4_025_5_III']=[1280,1573]    #wet from start    
#knots['T4_300_2_II']=[157,460]
#knots['T4_300_4_III']=[1272,1524]
#knots['T4_300_5_III']=[1377,1610]

excluded_samples=[
 'T3_100_4',        #FOV moved during acquisition
 'T3_300_6',        # water only in little pores
 'T3_300_3_III',    #no water
 'T4_025_3_III',    #wet from start
 'T3_300_8',        #FOV moved
 'T3_025_7',        #water only in small pores
 'T3_025_4_III',    #little water, data set incomplete, missing time steps reconstructed at PSI-ra-cluster, but only little water -> not reasonable to process?!
 'T3_025_3_II',      #little, water wet with acetone
 'T4_025_5_III',     #wet from start
 'T5_100_02_yarn_2'  #some strange error
 ]


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

stage = '02_pystack_registered'

for sample in samples:
    # if not sample == 'T3_025_9_III': continue
    # if not sample in repeats: continue
    if sample in excluded_samples: continue
    # if not sample[1]=='3': continue
#    if sample == 'T4_025_3_III': continue
    print(sample)
    if sample[1]=='4' or sample[1] == '_':
        z_low,z_high=knots[sample]
    targetFolder = os.path.join(baseFolder,sample,"01a_weka_segmented_dry")
#    targetFolder = os.path.join(baseFolder,sample,"01b_weka_segmented_mean")
    Folder = os.path.join(targetFolder,'temp')
    
    
    
    rawFolder=os.path.join(os.path.join(baseFolder,sample,stage),os.listdir(os.path.join(baseFolder,sample,stage))[0])
#    rawFolder=os.path.join(os.path.join(baseFolder,sample,'01_intcorrect_med'),os.listdir(os.path.join(baseFolder,sample,'01_intcorrect_med'))[0])
    names = os.listdir(rawFolder)
    if 'Thumbs.db' in names: names.remove('Thumbs.db')
    
    Stack = io.imread(os.path.join(Folder,'wekastack.tif'))
    
    folder2 = ''.join([rawFolder,'_2'])
    folder3 = ''.join([rawFolder,'_3'])
    
    if os.path.exists(folder3):
        print('several folders')
        Stack2 = io.imread(os.path.join(Folder,'wekastack2.tif'))
        Stack3 = io.imread(os.path.join(Folder,'wekastack3.tif'))
        Stack = np.concatenate((Stack3, Stack2, Stack), axis = 0)
        Stack2 = None
        Stack3 = None
    
    elif os.path.exists(folder2):
        print('several folders')
        Stack2 = io.imread(os.path.join(Folder,'wekastack2.tif'))
        Stack=np.concatenate((Stack2,Stack), axis=0)
        Stack2=None
        
    
    Stack = Stack[:,1,:,:]
    Stack = np.transpose(Stack, (1,2,0))
    
    if sample[1]=='4' or sample[1] == '_':
        
#        z_low,z_high=knot_boundary(sample)
        x_Folder=os.path.join(Folder,'x_set')
        y_Folder=os.path.join(Folder,'y_set')
        
        x_Stack = io.imread(os.path.join(x_Folder,'wekastack.tif'))
        x_Stack = x_Stack[:,1,:,:]
        
        y_Stack = io.imread(os.path.join(y_Folder,'wekastack.tif'))
        y_Stack = y_Stack[:,1,:,:]
        
        #retransformation of x,y, and z stack, tested, don't change
        x_Stack=np.transpose(x_Stack,(0,2,1))
        y_Stack=np.transpose(y_Stack,(1,0,2))
        
        Stack[:,:,z_low:z_high]=(Stack[:,:,z_low:z_high]+x_Stack+y_Stack)/3
    
    binStack=np.zeros(Stack.shape,dtype=np.uint8)
    binStack[np.where(Stack>confidence_value)]=255
    
    robpylib.CommonFunctions.ImportExport.WriteStackNew(os.path.join(targetFolder,'classified'),names,binStack)
    
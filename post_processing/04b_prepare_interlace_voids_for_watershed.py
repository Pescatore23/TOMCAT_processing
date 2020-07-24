# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 09:34:55 2019

@author: firo
"""

import robpylib
import os


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


baseFolder = r"A:\Robert_TOMCAT_4"

processingstage = '04a_void_space'
topfolder = '04b_split_void'



samples = os.listdir(baseFolder)

for sample in samples:
    if not sample[1]=='4': continue
    if not sample == 'T4_300_5': continue
    
    if not os.path.exists(os.path.join(baseFolder, sample, topfolder)):
        os.mkdir(os.path.join(baseFolder, sample, topfolder))
        
    Stack, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, processingstage))
    
    limit1 = knots[sample][0]
    limit2 = knots[sample][1]
    
    target1 = os.path.join(baseFolder, sample, topfolder, 'part_1')
    
    if not os.path.exists(target1):
        os.mkdir(target1)
        
    robpylib.CommonFunctions.ImportExport.WriteStackNew(target1, names[:limit1], Stack[:,:,:limit1])
    
    
    target2 = os.path.join(baseFolder, sample, topfolder, 'part_2')
    
    if not os.path.exists(target2):
        os.mkdir(target2)
        
    robpylib.CommonFunctions.ImportExport.WriteStackNew(target2, names[limit1:limit2], Stack[:,:,limit1:limit2])
    
    target3 = os.path.join(baseFolder, sample, topfolder, 'part_3')
    
    if not os.path.exists(target3):
        os.mkdir(target3)
        
    robpylib.CommonFunctions.ImportExport.WriteStackNew(target3, names[limit2:], Stack[:,:,limit2:])  
    
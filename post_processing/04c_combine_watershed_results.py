# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:53:18 2020

@author: firo
"""

import os
import robpylib

baseFolder = r"Z:\Robert_TOMCAT_4"

samples = os.listdir(baseFolder)
samples.sort()

# samples = ['T4_025_1_III']
parts = ['part_1', 'part_2', 'part_3']

for sample in samples:
    if sample == '.DS_Store': continue
    # already worked
    if sample in ['T4_025_1_III', 'T4_025_4', 'T4_100_2_III', 'T4_100_3', 'T4_300_1' ]: continue

#   some error
    if sample in ['T4_025_2_II']: continue

#  not segmented
    if sample in ['T4_300_4_III', 'T4_300_5']: continue
    print(sample)
    sourceFolder = os.path.join(baseFolder, sample, '05a_separated', 'disconnection=0.7 xsize=1.5 ysize=1.5 zsize=1 sigma=[] holes=0 particles=0 entire evaluation=[3D volumetric processing]')
    refFolder = os.path.join(baseFolder, sample, '04b_split_void')
    destFolder = os.path.join(sourceFolder, 'combined')
    print(destFolder)
    if not os.path.exists(destFolder):
        os.mkdir(destFolder)
    
    maximum = 0
    for part in parts:
        refnames = os.listdir(os.path.join(refFolder, part))
        if 'Thumbs.db' in refnames: refnames.remove('Thumbs.db')
        
        ims, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(sourceFolder, part))
        mask = ims==0
        ims = ims+maximum
        ims[mask]=0
        maximum = ims.max()
        robpylib.CommonFunctions.ImportExport.WriteStackNew(destFolder, refnames, ims)
        
        
    
    
    
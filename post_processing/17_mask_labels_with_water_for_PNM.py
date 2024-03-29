# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:46:02 2020

@author: firo
"""
import numpy as np
import xarray as xr
import os

sourceFolder = r"A:\Robert_TOMCAT_3b_netcdf4"
# sourceFolder2 = r"A:\Robert_TOMCAT_3_for_PNM"
destFolder = r"A:\Robert_TOMCAT_3b_netcdf4_for_PNM"
if not os.path.exists(destFolder):
    os.mkdir(destFolder)
# baseFolder = r"A:\Robert_TOMCAT_3"


# TODO: make different mask, this one does not work
for sample in os.listdir(sourceFolder):
    if sample[:3] == 'dyn':
        data = xr.load_dataset(os.path.join(sourceFolder, sample))
        name = data.attrs['name']
        print(name)
        # th=10000
        
        # last = -1
        # if name not in robpylib.TOMCAT.INFO.time_limit.keys(): continue
        # if name in robpylib.TOMCAT.INFO.time_limit.keys():
        #     last = robpylib.TOMCAT.INFO.time_limit[name]
        # sampleFolder = os.path.join(baseFolder, name, '02_pystack_registered')
        # scan = os.listdir(sampleFolder)[last]
        
        # Stack, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(sampleFolder, scan))        
        
        # mask = Stack[:,:,:data['label_matrix'].shape[2]]>th
        # mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        mask = (data['transition_matrix']>0).astype(np.uint8)
        
        data['label_matrix']=data['label_matrix']*mask.data
        filename = os.path.join(destFolder, sample)
        data.to_netcdf(filename)
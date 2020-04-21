# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:07:40 2020

@author: firo
"""

import xarray as xr
import robpylib
import os
import numpy as np


baseFolder = r'/Users/firo/NAS/Robert_TOMCAT_3_netcdf4_archives/processed_1200_dry_seg_aniso_sep/dyn_data_1200_dry_seg_aniso_sep'
destination = r'/Users/firo/NAS/Robert_TOMCAT_3_netcdf4_archives/fiber_data'

samples = os.listdir(baseFolder)

for sample in samples:
    if not sample[:3] == 'dyn': continue
    
    sample_data = xr.load_dataset(os.path.join(baseFolder, sample))
    # pore_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['pore_props_', sample[9:]])))
    # FIXME load fiber images to get real void geometry
    
    name = sample_data.attrs['name']
    if not name in robpylib.TOMCAT.INFO.samples_to_repeat: continue
    print(name)
    if name == 'T3_025_1': continue

    fiberFolder = os.path.join(r'/Users/firo/NAS/Robert_TOMCAT_3', name, '01a_weka_segmented_dry', 'classified')
    
    if not os.path.exists(fiberFolder): 
        print('fiber tifs NOT found')
        continue
    print('fiber tifs found')
    fibers, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiberFolder)
    
    
    dataset = xr.Dataset({'fibers': (['x','y','z'], fibers)},
                         coords = {'x': np.arange(0,fibers.shape[0]),
                                   'y': np.arange(0,fibers.shape[1]),
                                   'z': np.arange(0,fibers.shape[2])})
    dataset.attrs = sample_data.attrs
    
    filename = ''.join(['fiber_data_',name,'.nc'])
    path = os.path.join(destination, filename)
    
    dataset.to_netcdf(path)
                                                  
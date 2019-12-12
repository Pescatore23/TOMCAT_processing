# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:37:58 2019

@author: firo
"""


import numpy as np
import xarray as xr
import robpylib
import os

vx = 2.75E-6#m
baseFolder = r'Z:\Robert_TOMCAT_3'
#baseFolder = r'X:\TOMCAT3_processing_1'
destination = r'Z:\Robert_TOMCAT_3_netcfd4_archives\02_registered'

samples = os.listdir(baseFolder)


for sample in samples:
    if not sample in robpylib.TOMCAT.INFO.samples_to_repeat: continue
    if sample[1] == '4':
        print('T4 does not fit in memory, skipping ...')
        continue
    print(sample)
    if os.path.exists(os.path.join(destination, ''.join(['registered', sample, '.nc']))):
        print('already transformed, skipping...')
        continue
    regFolder = os.path.join(baseFolder, sample, '02_pystack_registered_from_5')
    #load data    
    tension = np.uint16(sample[3:6])
    sample_id = np.uint8(sample[7])
    last_3 = sample[-3:]
    if last_3 == 'III':
        repeat = 3
    elif last_3 == '_II':
        repeat = 2
    else:
        repeat = 1
        
    if os.path.exists(regFolder):
        folders = os.listdir(regFolder)
        time_len = len(folders)
        test_folder = os.path.join(regFolder, folders[0])
        imgs = os.listdir(test_folder)
        if 'Thumbs.db' in imgs: imgs.remove('Thumbs.db')
        
        test_im_path = os.path.join(regFolder, folders[0], os.listdir(test_folder)[0])
        test_im = robpylib.CommonFunctions.ImportExport.ReadImage(test_im_path,'')
        
        matrix = np.zeros([time_len, test_im.shape[0], test_im.shape[1], len(imgs)], dtype=np.uint16)
        time = robpylib.TOMCAT.TIME.TIME[sample][-time_len:]
        
        
        for t in range(time_len):
            time_path = os.path.join(regFolder, folders[t])
            matrix[t, :, :, :], _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(time_path, filetype = np.uint16, track=False)
            
        dataset = xr.Dataset({'registered': (['time', 'x', 'y', 'z'], matrix)},
                              coords = {'time': time,
                                        'x': np.arange(matrix.shape[1])*vx,
                                        'y': np.arange(matrix.shape[2])*vx,
                                        'z': np.arange(matrix.shape[3])*vx},
                                        attrs = {'name': sample,
                                                 'sample_id': sample_id,
                                                 'repeat': repeat,
                                                 'voxel_size': '2.75 um',
                                                 'vx': 2.75E-6,
                                                 'description': 'registered tif stacks combined in one large file'})
        
        dataset['registered'].attrs['units'] = 'greyvalue (16bit)'
        dataset['time'].attrs['units'] = 's'
        dataset['x'].attrs['units'] = 'm'
        dataset['y'].attrs['units'] = 'm'
        dataset['z'].attrs['units'] = 'm'
        
        dataset.to_netcdf(os.path.join(destination, ''.join(['registered', sample, '.nc'])))
        dataset = None
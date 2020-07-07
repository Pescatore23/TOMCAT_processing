# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:09:07 2020

@author: firo
"""

import xarray as xr
import numpy as np
import os
import scipy as sp
import scipy.ndimage

sourceFolder = r"Z:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep"
destinationFolder = r"Z:\Robert_TOMCAT_3_netcdf4_archives\expandedlabels"
if not os.path.exists(destinationFolder):
    os.mkdir(destinationFolder)
samples = os.listdir(sourceFolder)




sample = 'dyn_data_T3_025_3_III.nc'

data = xr.load_dataset(os.path.join(sourceFolder, sample))

label_matrix = data['label_matrix'].data


mask = label_matrix > 0

COM = sp.ndimage.center_of_mass(mask)
x0 = int(np.round(COM[0]))
y0 = int(np.round(COM[1]))


a = 45
size = 10
flag = False
name = ''.join([data.attrs['name'],'_size_',str(size)])
new_label = label_matrix[x0-a:x0+a,y0-a:y0+a,:]

for i in range(size):
    
    if flag:
        flip = np.flip(new_label, axis = 1).copy()
        mask = flip>0
        flip[mask] = flip[mask]+flip.max()
        new_label = np.concatenate((new_label, flip), axis = 1)
        flag = False
    else:
        flip = np.flip(new_label, axis = 0).copy()
        mask = flip>0
        flip[mask] = flip[mask]+flip.max()
        new_label = np.concatenate((new_label, flip), axis = 0)
        flag = True        


new_data = xr.Dataset({'label_matrix': (['x','y','z'], new_label)},
                      coords = {'x': np.arange(new_label.shape[0]),
                                'y': np.arange(new_label.shape[1]),
                                'z': np.arange(new_label.shape[2]),
                                'label': np.unique(new_label)[1:]})
new_data.attrs = data.attrs
new_data.attrs['name'] = name
new_data.attrs['size_factor'] = size
new_data.attrs['COM'] = COM
new_data.attrs['a'] = a

filename = ''.join(['dyn_',name,'_size_',str(size),'_a_',str(a),'.nc'])

new_data.to_netcdf(os.path.join(destinationFolder, filename))

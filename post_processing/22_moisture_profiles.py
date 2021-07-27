# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:33:00 2021

@author: firo
"""

import h5py
import os
import xarray as xr
import numpy as np

px = 2.75E-6
vx = px**3

baseFolder = '/home/firo/NAS/Robert_TOMCAT_4_netcdf4_split_v2'

file_list = os.listdir(baseFolder)

height = px*(2016-np.arange(2016))

profiles = np.zeros((2016,271,12))
samples = []

cc = 0

tflag = False

for file in file_list:
    if file[:3] == 'dyn':
        datafile = h5py.File(os.path.join(baseFolder, file))
        time = np.array(datafile['time'])
        transitions = np.array(datafile['transition_matrix'])
        time_steps = np.unique(transitions)[1:]
        
        name = datafile.attrs['name'].decode()
        
        sample_profiles = np.zeros((2016,271))
        any_water = transitions>0
        
        for ts in range(1,271):
            if ts in time_steps:
                water = np.bitwise_and(any_water,(transitions<ts+1))
                profile = np.count_nonzero(water, axis = (0,1))
                sample_profiles[:,ts] = profile
            else:
                sample_profiles[:,ts] = sample_profiles[:,ts-1]
        
        # profiles[:,:,cc] = sample_profiles
        cc = cc + 1

        data = xr.Dataset({'moisture_profile': (['height', 'time'], profiles*vx)},
                          coords = {'height': height,
                                   'time': time})
        
        data['height'].attrs['units'] = 'm'
        data['time'].attrs['units'] = 's'
        data.attrs['name'] = name
        
        destination = os.path.join(baseFolder, ''.join([name,'moisture_profiles.nc']))
        data.to_netcdf(destination)
        
        
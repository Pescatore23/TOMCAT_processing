# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:38:29 2019

@author: firo
"""

import os
import numpy as np
import xarray as xr
import robpylib
import matplotlib.pyplot as plt


rho = 997 #kg/m3
vx = 2.75E-6 #m
gamma = 72.7E-3 #N/m
theta = 50 #° plasma treated PET

drive = '//152.88.86.87/data118'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1200_dry_seg_aniso_sep'


sourceFolder = os.path.join(data_path, processing_version)

samples = os.listdir(sourceFolder)
samples = [samples[1]]

for sample in samples:
    if not sample[:3] == 'dyn': continue
    path = os.path.join(sourceFolder, sample)
    data = xr.load_dataset(path)
    name = data.attrs['name']
    label_matrix = data['label_matrix'].data
    labels = data['label'].data
    transition_matrix = data['transition_matrix'].data
    energy_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['energy_data_',name,'.nc'])))
    
    
    # clean label_matrix from irrelevant labels (noise)
    for i in range(1, label_matrix.max()+1):
        if not i in labels:
            label_matrix[label_matrix == i] = 0
    
#    directed_network, parameters = robpylib.CommonFunctions.pore_network.extract_throat_list(label_matrix, labels)
#    undirected_network = robpylib.CommonFunctions.pore_network.make_network_undirected(directed_network)
    
#    activation_ts = np.zeros(undirected_network.shape[0], dtype=np.uint16)
#    
#    for i in range(len(activation_ts)):
#        X = int(undirected_network[i, 10])
#        Y = int(undirected_network[i, 11])
#        Z = int(undirected_network[i, 12])
#        test_vol = transition_matrix[X-1:X+2,Y-1:Y+2,Z-1:Z+2]
#        activation_ts[i] = int(np.median(test_vol[test_vol>0]))
    time = data['time'].data
    step_size = time.copy()
    step_size[1:] = np.diff(time)
    
    R_beta = -4*energy_data['Helmholtz_energy'][:,-1].data/data['sig_fit_data'][:,2].data**2/vx**6/data['sig_fit_data'][:,1].data
    R_energy = -energy_data['diff_F']/step_size[None,1:]/data['filling'][:,:-1]**2/vx**6
    
#    FIXME: it is necessary to scale filling and diff_F with time step size, check if implemented in 05_(yes) and 10_(no)
    
    plt.figure()
    plt.plot(data['sig_fit_data'][:,0].data, R_beta,'.')
    plt.xlim(0, data['time'][-1])
    


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:09:34 2021

@author: firo
"""

import h5py
import os
import numpy as np
import json


baseFolder = r"T:\disk1\T5"
targetFolder = r"X:\Robert_TOMCAT_5_split"

samples = os.listdir(baseFolder)
samples = os.listdir(targetFolder)


TIME3 = {}

for sample in samples:
    sample_id = sample
    if sample[-6:-2] == 'yarn':
        sample_id = sample[:-7]
    metapath = os.path.join(baseFolder, sample_id, ''.join([sample_id,'_config.json']))
    datapath = os.path.join(baseFolder, sample_id, ''.join([sample_id,'.h5']))
    
    metadata = json.load(open(metapath,'r'))
    datafile = h5py.File(datapath, 'r')
    data = datafile['measurement']['instrument']['acquisition']['data']
    
    n = metadata['nimages']
    
    data = np.array(data)
    time_stamps = np.zeros(data.size)
    
    for i in range(data.size):
        time_stamps[i] = data[i][-2]
        
    TIME3[sample] = time_stamps[::n]/1e7
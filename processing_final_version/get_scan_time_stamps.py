# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:09:34 2021

@author: firo
"""

import h5py
import os
import numpy as np
import json
import pickle

baseFolder = r"/Volumes/Volume/disk1_2"
targetFolder = r"/Users/robfisch"

samples = os.listdir(baseFolder)
# samples = os.listdir(targetFolder)


TIME = {}

for sample in samples:
    print(sample)
    sample_id = sample
    if sample[-6:-2] == 'yarn':
        sample_id = sample[:-7]
    metapath = os.path.join(baseFolder, sample_id, ''.join([sample_id,'_config.json']))
    datapath = os.path.join(baseFolder, sample_id, ''.join([sample_id,'.h5']))
    
    if  os.path.exists(metapath):
        metadata = json.load(open(metapath,'r'))
        n = metadata['nimages']
        
    else:
        print('config file not found, hard coded n=500')
        n = 500

    if not os.path.exists(datapath):
        print('h5 file not found')
        continue
    
    
    datafile = h5py.File(datapath, 'r')
    data = datafile['measurement']['instrument']['acquisition']['data']
      
    
    data = np.array(data)
    time_stamps = np.zeros(data.size)
    
    for i in range(data.size):
        time_stamps[i] = data[i][-2]
        
    TIME[sample] = time_stamps[::n]/1e7

dumpfile = os.path.join(targetFolder, 'T2_time_part_1.p')
pickle.dump(open(dumpfile, 'wb'), TIME)

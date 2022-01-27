# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:09:34 2021

@author: firo
"""

import h5py
import os
import numpy as np
import json


baseFolder = r"/Volumes/Volume/disk1_2"
targetFolder = r"/Users/robfisch/NAS"

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
    
    if not os.path.exists(metapath):
        print('config file not found')
        continue
    if not os.path.exists(datapath):
        print('h5 file not found')
        continue
    
    metadata = json.load(open(metapath,'r'))
    datafile = h5py.File(datapath, 'r')
    data = datafile['measurement']['instrument']['acquisition']['data']
    
    n = metadata['nimages']
    
    data = np.array(data)
    time_stamps = np.zeros(data.size)
    
    for i in range(data.size):
        time_stamps[i] = data[i][-2]
        
    TIME[sample] = time_stamps[::n]/1e7

dumpfile = os.path.join(targetFolder, 'T2_time_part_1.txt')
with open(dumpfile, 'w') as convert_file: 
     convert_file.write(json.dumps(TIME))
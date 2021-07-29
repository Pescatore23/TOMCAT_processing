# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 09:13:40 2021

@author: firo
"""
import os
import numpy as np
import robpylib
from joblib import Parallel, delayed
import pickle



drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_4')
temp_folder= r"Z:\users\firo\joblib_tmp"


samples = os.listdir(baseFolder)

if '.DS_Store' in samples:
    samples.remove('.DS_Store')
    
interlace_labels = {}
    
def function(sample):
    label_folder = os.path.join(baseFolder, sample,'05b_labels_split_v2')
    sample_folder = os.path.join(baseFolder, sample, '06c_yarn_labels') 
    source1 = os.path.join(sample_folder, 'yarn1')
    source2 = os.path.join(sample_folder, 'yarn1')
    target = os.path.join(sample_folder, 'interface_zone')
    if not os.path.exists(target):
        os.mkdir(target)
        
    yarn1, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(source1, track=False, filetype=np.uint8)
    yarn2, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(source2, track=False, filetype=np.uint8)
    interface = np.bitwise_and(yarn1>0, yarn2>0).astype(np.uint8)
    robpylib.CommonFunctions.ImportExport.WriteStackNew(target, names, interface)
    
    inter_labels = 0
    if os.path.exists(label_folder):
        labels, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(label_folder, track=False)
        inter_labels = np.unique(labels[interface])
    
    return sample, inter_labels

num_jobs = 4
results = Parallel(n_jobs=num_jobs, temp_folder=temp_folder)(delayed(function)(sample) for sample in samples)    
    
pickle.dump(results, open(os.path.join(baseFolder, 'interface_labels'), 'rb'))
    
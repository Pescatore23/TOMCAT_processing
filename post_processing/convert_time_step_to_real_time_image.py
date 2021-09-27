# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:16:16 2020

@author: firo
"""

import robpylib
import os
import numpy as np
import pickle

baseFolder = r"V:\Robert_TOMCAT_4"
samples = os.listdir(baseFolder)

TIME = pickle.load(open(r"H:\11_Essential_Data\03_TOMCAT\TIME.p",'rb'))

# sample = 'T3_300_8_III'

def function(sample, baseFolder = baseFolder):
    transition_folder = os.path.join(baseFolder,sample,'03_gradient_filtered_transitions')
    time_folder = os.path.join(baseFolder,sample,'03c_gradient_filtered_real_time')

    print(sample)
    
    if not os.path.exists(time_folder):
        Stack, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(transition_folder)
        os.mkdir(time_folder)
        
        time = TIME[sample]
        
        new_stack = np.zeros(Stack.shape, dtype=np.uint16)
        
        for ts in range(len(time)):
            new_stack[Stack==ts+1] = np.uint16(time[ts])
            
        robpylib.CommonFunctions.ImportExport.WriteStackNew(time_folder, names, new_stack)
    
for sample in samples:
    if sample == '.DS_Store': continue
    function(sample)
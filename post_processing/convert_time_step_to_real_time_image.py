# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:16:16 2020

@author: firo
"""

import robpylib
import os
import numpy as np

sample = 'T3_025_3_III'
transition_folder = os.path.join(r'Z:\Robert_TOMCAT_3',sample,'03_gradient_filtered_transitions')
time_folder = os.path.join(r'Z:\Robert_TOMCAT_3',sample,'03c_gradient_filtered_real_time')

Stack, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(transition_folder)

if not os.path.exists(time_folder):
    os.mkdir(time_folder)
    
time = robpylib.TOMCAT.TIME.TIME[sample]

new_stack = np.zeros(Stack.shape, dtype=np.uint16)

for ts in range(len(time)):
    new_stack[Stack==ts+1] = time[ts]
    
robpylib.CommonFunctions.ImportExport.WriteStackNew(time_folder, names, new_stack)
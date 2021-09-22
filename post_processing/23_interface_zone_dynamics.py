# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:14:22 2021

@author: firo
"""

import xarray as xr
import numpy as np
import robpylib
import os

knots={}
knots['T4_025_3_III']=[1244,1572]   #wet from start
knots['T4_100_2_III']=[1204,1450]
knots['T4_300_3_III']=[1431,1664]
knots['T4_025_3']=[1032,1362]
knots['T4_025_4']=[777,1020]
knots['T4_100_3']=[171,405]         #on second small external HDD (rearguard)
knots['T4_100_4']=[233,441]
knots['T4_100_5']=[987,1229]
knots['T4_300_1']=[571,837]         
knots['T4_300_5']=[581,815]
knots['T4_025_1_III']=[1398,1664]
knots['T4_025_2_II']=[149,369]
knots['T4_025_3']=[1026,1341]
knots['T4_025_4']=[794,1017]              #on small external HD (rearguard)
knots['T4_025_5_III']=[1280,1573]    #wet from start    
knots['T4_300_2_II']=[157,460]
knots['T4_300_4_III']=[1272,1524]
knots['T4_300_5_III']=[1377,1610]

NAS = r"Z:"
ncFolder = os.path.join(NAS,'Robert_TOMCAT_4_netcdf4_split_v2_no_pore_size_lim')
baseFolder = os.path.join(NAS, 'Robert_TOMCAT_4')

files = os.listdir(ncFolder)

samples = {}
for file in files:
    if not file[:3] == 'dyn': continue
    sample = file[9:-3]
    samples[sample] = file

    
def interface_dynamics(sample, baseFolder = baseFolder, ncFolder = ncFolder, knots=knots):
    dyn_data = xr.open_dataset(os.path.join(ncFolder, samples[sample]))
    transitions = dyn_data['transition_matrix'].data
    time  = dyn_data['time'].data
    total_fill = dyn_data['filling'].sum(dim='label')
    dyn_data.close()
    
    k1 = knots[sample][0]
    k2 = knots[sample][1]   
    transitions = transitions[:,:,k1:k2]
    
    yarn1, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '06c_yarn_labels','yarn1'))
    yarn1 = yarn1[:,:,k1:k2]
    yarn1_trans, yarn1_counts =  np.unique(transitions*yarn1, return_counts=True)
    
    yarn2, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '06c_yarn_labels','yarn2'))
    yarn2 = yarn2[:,:,k1:k2]
    yarn2_trans, yarn2_counts =  np.unique(transitions*yarn2, return_counts=True)
    
    yarn1_trans = yarn1_trans[1:]
    yarn1_counts = yarn1_counts[1:]
    yarn2_trans = yarn2_trans[1:]
    yarn2_counts = yarn2_counts[1:]
        
    interface, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '06c_yarn_labels','interface_zone'))
    interface = interface[:,:,k1:k2]
    names = names[k1:k2]
    interface_transitions = interface*transitions
    
    knot_trans, knot_counts = np.unique(transitions, return_counts=True)
    knot_trans = knot_trans[1:]
    knot_counts = knot_counts[1:]
    int_trans, int_counts = np.unique(interface_transitions, return_counts=True)
    int_trans = int_trans[1:]
    int_counts = int_counts[1:]   
    
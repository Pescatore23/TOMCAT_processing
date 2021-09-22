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
    
def convert_array(data_ref, data_array, time):
    new_array = np.zeros(len(time), dtype=np.uint16)
    new_array[data_ref] = data_array
    return new_array

def filling(array, time):
    unique = np.unique(array, return_counts=True)
    ts = unique[0][1:]
    counts = unique[1][1:]
    new_array = convert_array(ts, counts, time)
    return new_array

    
def interface_dynamics(sample, baseFolder = baseFolder, ncFolder = ncFolder, knots=knots, samples=samples):
    dyn_data = xr.open_dataset(os.path.join(ncFolder, samples[sample]))
    transitions = dyn_data['transition_matrix'].data
    time  = dyn_data['time'].data
    total = dyn_data['volume'].sum(dim='label').data
    attrs = dyn_data.attrs.copy()
    dyn_data.close()
    
    k1 = knots[sample][0]
    k2 = knots[sample][1]   
    transitions = transitions[:,:,k1:k2]
    
    yarn1, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '06c_yarn_labels','yarn1'))
    yarn1 = yarn1[:,:,k1:k2]
    yarn1_fill = filling(transitions*yarn1, time)
    
    yarn2, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '06c_yarn_labels','yarn2'))
    yarn2 = yarn2[:,:,k1:k2]
    yarn2_fill = filling(transitions*yarn2, time)
        
    interface, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '06c_yarn_labels','interface_zone'))
    interface = interface[:,:,k1:k2]
    names = names[k1:k2]
    
    interface_transitions = interface*transitions
    
    knot_fill = filling(transitions, time)
    int_fill = filling(interface_transitions, time)
    y1_int_fill = filling(interface_transitions*yarn1, time)
    y2_int_fill = filling(interface_transitions*yarn2, time)
    
    data = xr.Dataset({'top_yarn': (time, yarn1_fill.cumsum()),
                       'bottom_yarn': (time, yarn2_fill.cumsum()),
                       'interface': (time, int_fill.cumsum()),
                       'knot': (time, knot_fill.cumsum()),
                       'top_interface': (time, y1_int_fill.cumsum()),
                       'bottom_interface': (time, y2_int_fill.cumsum()),
                       'full_sample': (time,total)},
                       coords = {'time': time})
    variables = list(data.keys())
    
    for var in variables:
        data[var].attrs['units'] = 'px'
    data['time'].attrs['units'] = 's'
    data.attrs = attrs
    filename = ''.join(['knot_dyn_', sample, '.nc'])
    data.to_netcdf(os.path.join(ncFolder, filename))
    
for sample in samples.keys():
    interface_dynamics(sample)
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:55:54 2020

@author: firo
"""

import xarray as xr
import numpy as np
import os
import robpylib
import pandas as pd


drive = r"\\152.88.86.87\data118"

data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1200_dry_seg_aniso_sep'

sourceFolder  = os.path.join(data_path, processing_version)

def get_pore_statistics(sourceFolder):
    samples_crude = os.listdir(sourceFolder)
    
    samples = []

    for sample in samples_crude:
        if sample[:3] == 'dyn':
            samples.append(sample)
            
    data_dict = {}
    
    c=0
    c25 = 0
    c100 = 0
    c300 = 0
    for sample in samples:
        c=c+1
    #    if c>1: continue
        if sample == 'T3_025_1': continue
        time_list = robpylib.TOMCAT.TIME.TIME[sample[9:-3]]
        dyn_data = xr.load_dataset(os.path.join(sourceFolder, sample))
        pore_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['pore_props_', sample[9:]])))
        data = dyn_data.merge(pore_data)
    #    data['parameter'] = parameter
    #    data.coords['sig_fit_var'] = ['t0 [s]', 'beta [1_s]', 'alpha [vx]', 'R2']
        variables = list(data['parameter'].data) + list(data['fit_var'].data) + list(data['sig_fit_var'].data) + list(data['property'].data) + ['rel_slope']+ ['max_filling_time [s]'] + ['norm_aspect_ratio']
        
    
        test_data = np.concatenate([data['dynamics'], data['fit_data'], data['sig_fit_data'], data['value_properties'], ], axis = 1)
        
        fw_id = variables.index('final water volume [vx]')
        relevant_pores = np.where(test_data[:,fw_id]>40)[0] #filter out noise speckles, pores that only have a few spurious pixel from bad registration at step 5
        test_data = test_data[relevant_pores,:]
        
        
        sl_id = variables.index('slope')
        test_data = np.concatenate([test_data, np.array([test_data[:,sl_id]/test_data[:,fw_id],]).transpose()], axis = 1) # rel_slope = slope/volume
        
        tmax_id = variables.index('time step of max filling rate')
        steps = test_data[:,tmax_id].copy().astype(np.uint16)
        time_max = np.array([time_list[step] for step in steps])
        
        test_data = np.concatenate([test_data, np.array([time_max,]).transpose()], axis = 1)
        
        
        
        ma_id = variables.index("major_axis")
        area_id = variables.index("mean_area")
        
        norm_a_r = test_data[:,ma_id]/np.sqrt(test_data[:, area_id])
        test_data = np.concatenate([test_data, np.array([norm_a_r,]).transpose()], axis = 1)
        
        
    
        data_dict[sample[9:-3]] = test_data.copy()   
    
    #    all samples together    
        if c==1:
            combined_data = test_data
        if c>1:
            combined_data = np.concatenate([combined_data, test_data], axis = 0)
    ##########################
    
    
        # combined datasets per tension value
        if data.attrs['tension'] == 25:
            c25 = c25 + 1  
            if c25 == 1:
                combined_025 = test_data   
            if c25 > 1:
                combined_025 = np.concatenate([combined_025, test_data], axis = 0)
        
        if data.attrs['tension'] == 100:
            c100 = c100 + 1  
            if c100 == 1:
                combined_100 = test_data   
            if c100 > 1:
                combined_100 = np.concatenate([combined_100, test_data], axis = 0)
                
        if data.attrs['tension'] == 300:
            c300 = c300 + 1  
            if c300 == 1:
                combined_300 = test_data   
            if c300 > 1:
                combined_300 = np.concatenate([combined_300, test_data], axis = 0)
    #  samples         
    
    combined_data[np.isnan(combined_data)] = 0
    combined_data[np.isinf(combined_data)] = 0
    
    combined_025[np.isnan(combined_025)] = 0
    combined_025[np.isinf(combined_025)] = 0
    
    combined_100[np.isnan(combined_100)] = 0
    combined_100[np.isinf(combined_100)] = 0
    
    combined_300[np.isnan(combined_300)] = 0
    combined_300[np.isinf(combined_300)] = 0
    ylims = {'filling start time step': (0, 360),
             'filling end time step': (0, 360),
             'filling start [s]': (0, 1750),
             'filling end [s]': (0, 1750),
             'max filling rate [vx_s]': (0, 40000), #(0, 5E9),
             'time step of max filling rate': (0, 360),
             'median filling rate [vx_s]': (0, 20), #(0, 120),
             'num filling peaks': (0, 20), #(0, 35),
             'pore volume [vx]': (0, 1000000), # (0, 2500000),
             'slope': (0, 4000), #(0, 30000),
             'rel_slope': (0,0.025), #(0,1),
             'onset': (0, 1800), #(0, 5E18),
             't0 [s]': (0, 1750), # (-80000, 50000),
             'beta [1_s]': (0,4), # (0, 600),
             'alpha [vx]': (0, 350000), #(0, 7E9),
             'R2': (0, 1.05),
             'major_axis': (0, 1800),
             'minor_axis': (0, 170),
             'aspect_ratio': (0,50), #(0, 90),
             'median_shape_factor': (0, 0.08),
             'mean_shape_factor': (0, 0.08), #(0, 0.8),
             'shape_factor_std': (0, 0.02), #(0, 1.6),
             'shp_fac_rel_var': (0,1), #(0,3),
             'tilt_axis_state': (-0.05, 1.05),
             'tilt_angle_grad': (0, 30), #(0, 80),
             'eccentricity_from_vertical_axis': (0, 65),
             'eccentricity_from_tilt_axis': (0, 45),
             'distance_end_to_end': (0, 1500),
             'arc_length': (0, 1750),
             'tortuosity': (0.9, 1.3), #(0.95, 3),
             'volume': (0, 600000),
             'mean_area': (0, 500),# (0, 2750),
             'median_area': (0, 500), #(0,3000),
             'area_std': (0,500), #(0,1750),
             'area_rel_var': (0, 1.25),
             'max_filling_time [s]': (0, 1750),
             'norm_aspect_ratio': (0,200),
             'angular_dist': (0,100),#(0,360)
             'final_saturation': (0,1),
             'final water volume [vx]': (0, 200000),
             'shell_mobility_%': (0, 1.5)
             }
    
    ylim_frame = pd.DataFrame(ylims)
    ylim_ar = xr.DataArray(ylim_frame, dims=['limits', 'variables_limit'])
    
    
    dataset = xr.Dataset({'pore_data': (['all_pores','variables'], combined_data),
                         'pore_data_025': (['025_pores', 'variables'], combined_025),
                         'pore_data_100': (['100_pores', 'variables'], combined_100),
                         'pore_data_300': (['300_pores', 'variables'], combined_300),
                         'valuelimits': ylim_ar},
                         coords = {'variables': variables,
                                  'all_pores': np.arange(combined_data.shape[0]),
                                  '025_pores': np.arange(combined_025.shape[0]),
                                  '100_pores': np.arange(combined_100.shape[0]),
                                  '300_pores': np.arange(combined_300.shape[0])},
                         attrs = {'comment': 'per pore data for all samples merged according to their mechanical state disregarding individual samples'})
    
    filename = os.path.join(sourceFolder, 'per_pore_statistics.nc')
    dataset.to_netcdf(filename)
    
    # #values that cover whole dataset

    
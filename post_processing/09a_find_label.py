# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:28:06 2019

@author: firo
"""

import xarray as xr
import os
import numpy as np
import robpylib

good_samples = robpylib.TOMCAT.INFO.good_samples

property_1 = 'beta [1_s]'
prop_1_val = 4

property_2 = 'alpha [vx]'
prop_2_val = 40000


folder = r"W:\Robert_TOMCAT_3_netcdf4_archives\processed_1400_dry_seg_aniso_sep_all_samples"

sample_list = []
label_list = []

for sample in os.listdir(folder):
    test1 = []
    test2 = []
    if not sample[:3] == 'dyn': continue
    
    sample_data = xr.load_dataset(os.path.join(folder, sample))
    sample_name = sample_data.attrs['name']
    if not sample_name in good_samples: continue
    
    sig_fit_beta = sample_data['sig_fit_data'].data[:,1]
    
    test_1 = np.where(sig_fit_beta > prop_1_val)[0]
    
    if len(test_1) > 0:
        test_2 = np.where(sample_data['sig_fit_data'].data[:,2] > prop_2_val)[0]
        if len(test_2) > 0:
            for i in test_1:
                if i in test_2:
                    if sig_fit_beta[i] > 10: continue
                    if sample_data['sig_fit_data'].data[i,2] > 250000: continue
#                    print(sig_fit_beta[i])
#                    print(sample_data['sig_fit_data'].data[i,2])
                    sample_list.append(sample_name)
                    label_list.append(sample_data['label'][i].data)
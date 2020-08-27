# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:17:56 2020

@author: firo
"""
# data['dynamics'].sel(parameter = 'num filling peaks')

import numpy as np
import xarray as xr
import os
from scipy.interpolate import interp1d
import robpylib

ecdf = robpylib.CommonFunctions.Tools.weighted_ecdf

sourceFolder = r"Z:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep"

peak_num = np.array([])

for sample in os.listdir(sourceFolder):
    if sample[:3] == 'dyn':
        data = xr.load_dataset(os.path.join(sourceFolder, sample))
        num_sample = data['dynamics'].sel(parameter = 'num filling peaks').data
        peak_num = np.concatenate([peak_num, num_sample])
        
peak_num = peak_num[peak_num>1]

x, y = ecdf(peak_num)

peak_fun = interp1d(y, x, fill_value = 'extrapolate')

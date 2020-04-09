# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:38:08 2020

This is a horrible code. Don't judge me, it was meant to do the job

@author: firo
"""

import numpy as np
import scipy as sp
import scipy.signal
import xarray as xr
import os
from joblib import Parallel, delayed
import robpylib

num_cores = 10#mp.cpu_count()
drive = '//152.88.86.87/data118'
processing_version = 'processed_1200_dry_seg_aniso_sep'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives', processing_version)




def peak_diff_calc(file):    
    dyn_data = xr.load_dataset(file)
    sample_name = dyn_data.attrs['name']
    tension = dyn_data.attrs['tension']
    label_matrix = dyn_data['label_matrix'].data
    labels = dyn_data['label'].data
    throat_params = robpylib.CommonFunctions.pore_network.extract_throat_list(label_matrix, labels)
    time = dyn_data['time'].data
    peak_distances = -np.ones([len(dyn_data['label'].data),20])
    heights = peak_distances.copy()
    filling = dyn_data['filling'].data
    pore_peaks = []
    pore_heights = []
    for i in range(filling.shape[0]):
        peaks, props = sp.signal.find_peaks(filling[i,:], height = 500)
        pore_peaks.append(peaks)
        heights[i,:len(peaks)] = props['peak_heights']
        pore_heights.append(props['peak_heights'])
        diffs = np.diff(time[peaks])
        peak_distances[i,:len(diffs)] = diffs
    links = np.uint16(throat_params[:,:1])
    link_diff = np.zeros(len(links))
    link_diff_of_max = link_diff.copy()
    # compare first peaks and max peaks
    for c in range(len(links)):
        p1 = links[c,0]
        p2 = links[c,1]
        i1 = np.where(labels==p1)[0][0]
        i2 = np.where(labels==p2)[0][0]
        ts1 = pore_peaks[i1][0]
        ts2 = pore_peaks[i2][0]
        link_diff[c] = time[ts1] - time[ts2]
        mx1 = pore_peaks[i1][np.argmax(pore_heights[i1])]
        mx2 = pore_peaks[i2][np.argmax(pore_heights[i2])]
        link_diff_of_max[c] = time[mx1] - time[mx2]
    pore_diffs = np.concatenate((link_diff, link_diff_of_max), axis = 1)
    
    return peak_distances, sample_name, tension, heights, pore_diffs


c=0
files_crude = os.listdir(data_path)
files = []
for file in files_crude:
    if c>1: continue
    if file[:3] == 'dyn': 
        files.append(file)
        c=c+1

results = Parallel(n_jobs = num_cores)(delayed(peak_diff_calc) for file in files)

data = np.array([])
data_peaks = np.array([])
names = []
tensions = []
data_025 = np.array([])
data_100 = np.array([])
data_300 = np.array([])
data_025_peaks = np.array([])
data_100_peaks = np.array([])
data_300_peaks = np.array([])

diffs = np.array([])
diffs_025 = np.array([])
diffs_100 = np.array([])
diffs_300 = np.array([])

for result in results:
    data_res = np.array(result[0])
    data_peaks_res = np.array(result[3])
    diff_res = np.array(result[4])
    data = np.concatenate((data, data_res))
    diffs  = np.concatenate((diffs, diff_res))
    data_peaks = np.concatenate((data_peaks, data_peaks_res))
    names.append(result[1])
    tensions.append(result[2])
    
    if result[2] == 25:
        data_025 = np.concatenate((data_025, data_res))
        data_025_peaks = np.concatenate((data_025_peaks, data_peaks_res))
        diffs_025 = np.concatenate((diffs_025, diff_res))
    if result[2] == 100:
        data_100 = np.concatenate((data_100, data_res))
        data_100_peaks = np.concatenate((data_100_peaks, data_peaks_res))
        diffs_100 = np.concatenate((diffs_100, diff_res))
    if result[2] == 300:
        data_300 = np.concatenate((data_300, data_res))
        data_300_peaks = np.concatenate((data_300_peaks, data_peaks_res))
        diffs_300 = np.concatenate((diffs_300, diff_res))
    

dataset = {'tensions': tensions, 
           'names': names,
           'data': data,
           'data_heights': data_peaks,
           'diffs': diffs,
           'data_025': data_025,
           'data_100': data_100,
           'data_300': data_300,
           'data_025_heights': data_025_peaks,
           'data_100_heights': data_100_peaks,
           'data_300_heights': data_300_peaks,
           'diffs_025': diffs_025,
           'diffs_100': diffs_100,
           'diffs_300': diffs_300}

np.save(dataset, os.path.join(data_path, 'peak_diffs.npy'))

dataset = xr.Dataset(dataset)

dataset.to_netcdf(os.path.join(data_path, 'peak_diffs.nc'))


# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:58:02 2020

@author: firo
"""

import os
import xarray as xr
import numpy as np
import scipy as sp
from joblib import Parallel, delayed
import multiprocessing as mp
import robpylib
# from scipy import sparse
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt


drive = r'\\152.88.86.87\data118'
# drive = r"NAS"
# drive =  r'Z:\'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
# data_path = r'Z:\Robert_TOMCAT_3_netcdf4_archives'
processing_version = 'processed_1200_dry_seg_aniso_sep'
# folder = r'Z:\Robert_TOMCAT_3'

sourceFolder = os.path.join(data_path, processing_version)

heigth_crit = 150 #vx/s
dist_crit = 7 #s
prom_crit = 75 #vx/s

samples = os.listdir(sourceFolder)

def resample_data(data, time):
    spline = interp1d(time, data, fill_value = 'extrapolate')
    new_time = np.arange(time.min(),time.max(),1)
    new_data = spline(new_time)
    return new_time, new_data


def neigbour_flux_analysis_v2(conn, data, time):
    # returns the peaks of pore neighbor pairs
    volume = data['volume'].sel(label=conn[0]).data + data['volume'].sel(label=conn[1]).data
    flux = np.gradient(volume, time)
    new_time, new_flux = resample_data(flux, time)
    peaks, props = sp.signal.find_peaks(new_flux, height=heigth_crit, distance=dist_crit, prominence=prom_crit)
    return peaks, props
    

def diffs_v1(conn, data):
    # gives the difference of the sigmoid t0 between neighbor pairs
    t1 = data['sig_fit_data'].sel(label=conn[0], sig_fit_var='t0 [s]')
    t2 = data['sig_fit_data'].sel(label=conn[1], sig_fit_var='t0 [s]')  
    if t2>0 and t1>0:
        dt = t2-t1
    else:
        dt=-1
    return conn[0], conn[1], dt

def diffs_v2(conn, data, time):
    peaks, props = neigbour_flux_analysis_v2(conn, data, time)
    dt = np.diff(peaks)
    peak_heights = props['peak_heights']
    return conn[0], conn[1], dt, peaks, peak_heights

def diffs_v3(label, adj_mat, data, time):
    nb = np.where(adj_mat[label,:])[0]
    k = len(nb)-1
    volume = data['volume'].sel(label=nb).sum(axis=0).data
    flux = np.gradient(volume, time)
    new_time, new_flux = resample_data(flux, time)
    peaks, props = sp.signal.find_peaks(new_flux, height=heigth_crit, distance=dist_crit, prominence=prom_crit)
    dt = np.diff(peaks)
    peak_heights = props['peak_heights']
    return label, k, dt, peaks, peak_heights
    

def diffs_v4(label, data, time):
    # returns the peak distances inside a pore
    volume = data['volume'].sel(label=label).data
    flux = np.gradient(volume, time)
    new_time, new_flux = resample_data(flux, time)
    peaks, props = sp.signal.find_peaks(new_flux,  height=heigth_crit, distance=dist_crit, prominence=prom_crit)
    dt = np.diff(peaks)
    peak_heights = props['peak_heights']
    return label, dt, peaks, peak_heights


def __main__(sample):
    if sample[:3] == 'dyn':
        sample_data = xr.load_dataset(os.path.join(sourceFolder, sample))
        name = sample_data.attrs['name']
        tension = sample_data.attrs['tension']
        time = sample_data['time']
        labels = sample_data['label'].data
        label_mat = sample_data['label_matrix'].data
        adj_mat = robpylib.CommonFunctions.pore_network.adjacency_matrix(label_mat)
        mask = np.ones(adj_mat.shape[0], np.bool)
        mask[labels] = False
        adj_mat[mask,:] = False
        adj_mat[:,mask] = False 
        adj_sparse = sp.sparse.coo_matrix(adj_mat)
        conn_list = zip(adj_sparse.row, adj_sparse.col)
        conn_list = list(conn_list)
        
        diff_data_v1 = np.zeros((3,len(conn_list)))-1
        
        diff_data_v2 = np.zeros((30, len(conn_list)))-1
        peak_data_v2 = diff_data_v2.copy()
        peak_height_data_v2 = diff_data_v2.copy()
        
        diff_data_v3 = np.zeros((30, len(labels)))-1
        diff_data_v4 = diff_data_v3.copy()
        
        peak_data_v3 = diff_data_v4.copy()
        peak_data_v4 =  diff_data_v4.copy()
        
        peak_height_data_v3 = diff_data_v4.copy()
        peak_height_data_v4 = diff_data_v4.copy()
        
        
        c=0
        for conn in conn_list:
            p1,p2, dt = diffs_v1(conn, sample_data)
            diff_data_v1[:,c] = [p1,p2,dt]
            
            p1, p2, dt, peaks, peak_heights = diffs_v2(conn, sample_data, time)
            peak_sorting = np.argsort(peak_heights)[-28:]
            diff_sorting = np.argsort(dt)[-28:]
            
            diff_data_v2[0,c] = p1
            diff_data_v2[1,c] = p2
            diff_data_v2[2:len(diff_sorting)+2,c] = dt[diff_sorting]
            
            peak_data_v2[0,c] = p1
            peak_data_v2[1,c] = p2
            peak_data_v2[2:len(peak_sorting)+2,c] = peaks[peak_sorting]
                      
            peak_height_data_v2[:2,c] = [p1,p2]
            peak_height_data_v2[2:len(peak_sorting)+2,c] = peak_heights[peak_sorting]
            c=c+1
            
        c= 0  
        for label in labels:
            # v3
            _, k, dt, peaks, peak_heights = diffs_v3(label, adj_mat, sample_data, time)
            peak_sorting = np.argsort(peak_heights)[-29:]
            diff_sorting = np.argsort(dt)[-29:]   
            
            diff_data_v3[0,c] = label
            diff_data_v3[1,c] = k
            diff_data_v3[2:len(diff_sorting)+2, c] =dt[diff_sorting]
            
            peak_data_v3[0,c] = label
            peak_data_v3[1,c] = k
            peak_data_v3[2:len(peak_sorting)+2, c] = peaks[peak_sorting]
            
            peak_height_data_v3[0,c] = label
            peak_height_data_v3[1,c] = k
            peak_height_data_v3[2:len(peak_sorting)+2, c] = peak_heights[peak_sorting]
            
            # v4
            _, dt, peaks, peak_heights = diffs_v4(label, sample_data, time)
            
            peak_sorting = np.argsort(peak_heights)[-29:]
            diff_sorting = np.argsort(dt)[-29:]
            diff_data_v4[0,c] = label
            diff_data_v4[1,c] = k
            diff_data_v4[2:len(diff_sorting)+2, c] = dt[diff_sorting]
            
            peak_data_v4[0,c] = label
            peak_data_v4[1,c] = k
            peak_data_v4[2:len(peak_sorting)+2, c] = peaks[peak_sorting]
            
            peak_height_data_v4[0,c] = label
            peak_height_data_v4[1,c] = k
            peak_height_data_v4[2:len(peak_sorting)+2,c] = peak_heights[peak_sorting]
            
            c=c+1
            
        # data_dict={}
        
        # data_dict['diffs_v1'] = diff_data_v1
        # data_dict['diffs_v2'] = diff_data_v2
        # data_dict['diffs_v3'] = diff_data_v3
        # data_dict['diffs_v4'] = diff_data_v4
        
        # data_dict['peaks_v2'] = peak_data_v3
        # data_dict['peaks_v3'] = peak_data_v3
        # data_dict['peaks_v4'] = peak_data_v4
        
        # data_dict['peak_heights_v2'] = peak_height_data_v2
        # data_dict['peak_heights_v3'] = peak_height_data_v3
        # data_dict['peak_heights_v4'] = peak_height_data_v4
        
        # data_dict = xr.Dataset(data_dict)
        
        
        data_dict = xr.Dataset({'diffs_v1': (['ax_v1_0', 'pair'], diff_data_v1),
                                'diffs_v2': (['ax_0', 'pair'], diff_data_v2),
                                'peaks_v2': (['ax_0', 'pair'], peak_data_v2),
                                'peak_heights_v2': (['ax_0', 'pair'], peak_height_data_v2),
                                'diffs_v3': (['ax_0', 'label'], diff_data_v3),
                                'diffs_v4': (['ax_0', 'label'], diff_data_v4),
                                'peaks_v3': (['ax_0', 'label'], peak_data_v3),
                                'peaks_v4': (['ax_0', 'label'], peak_data_v4),
                                'peak_heights_v3': (['ax_0', 'label'], peak_height_data_v3),
                                'peak_heights_v4': (['ax_0', 'label'], peak_height_data_v4),
                                },
                               coords= {'ax_v1_0': np.arange(diff_data_v1.shape[0]),
                                        'pair': np.arange(diff_data_v1.shape[1]),
                                        'ax_0': np.arange(diff_data_v2.shape[0]),
                                        'label': labels})
                                                          
        
        data_dict.attrs['name'] = name
        data_dict.attrs['tension'] = tension
        data_dict.attrs['comment'] = 'comparison of different waiting time and filling peaks extraction methods'
        
        data_dict.attrs['v1'] = 'difference of sigmoid fit t0 of neighbor pores'
        data_dict.attrs['v2'] = 'peaks obtained from combined filling curve of neighbor pairs'
        data_dict.attrs['v3'] = 'peaks obtained from combined filling curve of one pore and all its neighbors'
        data_dict.attrs['v4'] = 'peaks obtained from filling curve of one pore as reference'
        data_dict.attrs['v1_v2_data_structure'] = '0: pore 1, 1: pore 2, rest: data'
        data_dict.attrs['v3_v4_data_structure'] = '0: pore, 1: degree/# neighbors, rest: data'
        
        filename = os.path.join(sourceFolder, ''.join(['peak_diff_data_', name,'.nc']))
        data_dict.to_netcdf(filename)
            
            
            
        
result = Parallel(n_jobs= 20)(delayed(__main__)(sample) for sample in samples)      
        
# do a separate script or add for energy difference before peaks
# problem: you need energy after first peak and before second peak
        
        

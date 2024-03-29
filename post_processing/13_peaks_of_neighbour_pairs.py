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
# import robpylib
# from scipy import sparse
from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
num_cores = 8
import socket
host = socket.gethostname()
import networkx as nx


# drive = r'\\152.88.86.87\data118'
# drive = r"NAS2"
drive =  r"Z:"
temp_folder = None
if host == 'DDM06609': 
    drive = r"A:"
    temp_folder = r"Z:\users\firo\joblib_tmp"
if host == 'ddm06608': 
    drive = r"V:"
    temp_folder = r"Z:\users\firo\joblib_tmp"
    
# data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
# data_path = os.path.join(drive, 'Robert_TOMCAT_5_netcdf4') #TODO: change also peak definition fot T4!!
# data_path = r'Z:\Robert_TOMCAT_3b_netcdf4'
data_path = os.path.join(drive, 'Robert_TOMCAT_4_netcdf4_split_v2_no_pore_size_lim')
# data_path = r'B:\Robert_TOMCAT_5_netcdf4'
# processing_version = 'processed_1200_dry_seg_aniso_sep'
# processing_version = 'for_PNM'
# folder = r'Z:\Robert_TOMCAT_3'

# sourceFolder = os.path.join(data_path, processing_version)
sourceFolder = data_path

# TODO: revise peak heights for TOMCAT4

heigth_crit = 300 #vx/s 300 for T4
dist_crit = 1#s 7 #s 1s for TOMCAT_4 and 7s for TOMCAT 5
prom_crit = 150 #vx/s 150 for T4
sampling = 0.1# resample data to 4x maximum temporal resolution for equidistant time everywhere: T3:1s, T4:0.2s

samples = os.listdir(sourceFolder)

def resample_data(data, time, sampling=sampling):
    spline = interp1d(time, data, fill_value = 'extrapolate')
    new_time = np.arange(time.min(),time.max(),sampling)
    new_data = spline(new_time)
    return new_time, new_data


def neigbour_flux_analysis_v2(conn, data, time, sampling = sampling):
    # returns the peaks of pore neighbor pairs
    volume = data['volume'].sel(label=conn[0]).data + data['volume'].sel(label=conn[1]).data
    flux = np.gradient(volume, time)
    new_time, new_flux = resample_data(flux, time)
    peaks, props = sp.signal.find_peaks(new_flux, height=heigth_crit, distance=dist_crit/sampling, prominence=prom_crit)
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

def diffs_v2(conn, data, time, sampling = sampling):
    peaks, props = neigbour_flux_analysis_v2(conn, data, time)
    dt = np.diff(peaks)*sampling
    peak_heights = props['peak_heights']
    return conn[0], conn[1], dt, peaks, peak_heights

def diffs_v3(label, adj_mat, data, time, sampling = sampling):
    nb = np.where(adj_mat[label,:])[0]
    k = len(nb)-1
    volume = data['volume'].sel(label=nb).sum(axis=0).data
    flux = np.gradient(volume, time)
    new_time, new_flux = resample_data(flux, time)
    peaks, props = sp.signal.find_peaks(new_flux, height=heigth_crit, distance=dist_crit/sampling, prominence=prom_crit)
    dt = np.diff(peaks)*sampling
    peak_heights = props['peak_heights']
    return label, k, dt, peaks, peak_heights
    

def diffs_v4(label, data, time, sampling = sampling):
    # returns the peak distances inside a pore
    volume = data['volume'].sel(label=label).data
    flux = np.gradient(volume, time)
    new_time, new_flux = resample_data(flux, time)
    peaks, props = sp.signal.find_peaks(new_flux,  height=heigth_crit, distance=dist_crit/sampling, prominence=prom_crit)
    dt = np.diff(peaks)*sampling
    peak_heights = props['peak_heights']
    return label, dt, peaks, peak_heights

def reconstruct_adj_matrix(old_diff_data):
    conns = old_diff_data['pairs'].data
    conn_list = list(conns)
    size = conns.max()+1
    adj_mat = np.zeros([size,size], dtype=np.bool)
 
    for conn in conn_list:
        adj_mat[conn[0], conn[1]] = True
    
    # make sure that matrix is symmetric (as it should be)
    adj_mat = np.maximum(adj_mat, adj_mat.transpose())
    
    return adj_mat, conn_list

def reconstruct_graph_from_netcdf4(path):
    data = xr.load_dataset(path)
    nodes = data['nodes'].data
    mapping = {}
    for i in range(len(nodes)):
        mapping[i] = nodes[i]
    adj_matrix = data['adj_matrix'].data
    graph = nx.from_numpy_array(adj_matrix)
    H = nx.relabel_nodes(graph, mapping)
    return H
    
def __main__(sample):
    if sample[:3] == 'dyn':
        sample_data = xr.open_dataset(os.path.join(sourceFolder, sample))
        name = sample_data.attrs['name']
        filename = os.path.join(sourceFolder, ''.join(['peak_diff_data_', name,'.nc']))
        if not os.path.exists(filename):
            tension = sample_data.attrs['tension']
            time = sample_data['time'].data
            labels = sample_data['label'].data
            # old_file = os.path.join(sourceFolder, 'resampled_to_1s_too_crude', ''.join(['peak_diff_data_', name,'.nc']))
            
            
            # if os.path.exists(old_file ):
            #     old_diff_data = xr.load_dataset(old_file)
            #     adj_mat, conn_list = reconstruct_adj_matrix(old_diff_data)
            #     mask = np.ones(adj_mat.shape[0], np.bool)
            #     mask[labels] = False
            #     adj_mat[mask,:] = False
            #     adj_mat[:,mask] = False 
            
            # else:
            #     label_mat = sample_data['label_matrix'].data
            #     adj_mat = robpylib.CommonFunctions.pore_network.adjacency_matrix(label_mat, num_cores=8)
            gpath = os.path.join(sourceFolder, ''.join(['network_',sample,'.nc']))
            graph = reconstruct_graph_from_netcdf4(gpath)
            adj_mat = nx.to_numpy(graph)
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
            
            conn_list = np.array(conn_list)
            c= 0  
            for label in labels:
                # v3
                _, k, dt, peaks, peak_heights = diffs_v3(label, adj_mat, sample_data, time)
                peak_sorting = np.argsort(peak_heights)[-28:]
                diff_sorting = np.argsort(dt)[-28:]   
                
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
                                    'pairs': (['pair', 'connection'], conn_list)
                                    },
                                   coords= {'ax_v1_0': np.arange(diff_data_v1.shape[0]),
                                            'pair': np.arange(diff_data_v1.shape[1]),
                                            'ax_0': np.arange(diff_data_v2.shape[0]),
                                            'label': labels,
                                            'connection': np.array([0,1])})
                                                              
            
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
        sample_data.close()
            
            
            
        
result = Parallel(n_jobs= num_cores, temp_folder=temp_folder)(delayed(__main__)(sample) for sample in samples)      

# for sample in samples:
#     print(sample)
#     __main__(sample)
        
# do a separate script or add for energy difference before peaks
# problem: you need energy after first peak and before second peak
        
        

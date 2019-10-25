# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:57:31 2019

@author: firo
"""
#import sys
#library=r"R:\Scratch\305\_Robert\Python_Library"
#
#if library not in sys.path:
#    sys.path.append(library)
    
    
import xarray as xr
import os
from scipy import ndimage
#import RobPyLib
import numpy as np
#import skimage
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt


num_cores = mp.cpu_count()


rho = 997 #kg/m3
vx = 2.75E-6 #m
gamma = 72.7E-3 # N/m
theta = 50 #° plasma treated PET

drive = '//152.88.86.87/data118'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1400_dry_seg_aniso_sep'

sourceFolder  = os.path.join(data_path, processing_version)

samples = os.listdir(sourceFolder)



def helmholtz_energy(A_int, Awet, gamma = gamma, theta = theta/180*np.pi):
    dF = gamma*(np.diff(A_int, axis = 1) - np.cos(theta)*np.diff(A_wet, axis = 1))
    return dF


def interface_checker(im, interface, t, radius=1):
    indices = np.array(np.nonzero(interface))
    num_ind = indices.shape[1]
    if num_ind>0:
        for i in range(num_ind):
            index = indices[:,i]
            env = im[index[0]-radius:index[0]+radius+1, index[1]-radius:index[1]+radius+1, index[2]-radius:index[2]+radius+1]
            if env.size<1:
                interface[index[0],index[1],index[2]]=False
            else:
                maximum = env.max()
                if maximum < t: interface[index[0],index[1],index[2]]=False
    return interface


def pore_interface_tracking(label, label_matrix, transitions, time):
    #pore = 31
    #label = pore+1
    A_int = np.zeros(len(time))
    A_wet = np.zeros(len(time))
    mask=label_matrix==label
    
    bounding_box = ndimage.find_objects(mask)[0]
    
    transition = transitions[bounding_box].data
    mask = mask[bounding_box]
    pore_filling = transition*mask
    watermask = pore_filling>0
    
    
#    interface = np.zeros([len(time), watermask.shape[0], watermask.shape[1], watermask.shape[2]], dtype=np.bool)
    
    for t in range(len(time)):
        
    #    t = tt + 40   #change accordingly to interval of pore filling
    
        water = watermask*(pore_filling<t)
#            water_interface = RobPyLib.CommonFunctions.Tools.interface_tracker(water, dist=1)
        dt = ndimage.distance_transform_cdt(water)
        dt_inv = ndimage.distance_transform_cdt(~water)
        water_interface = dt==1
        Area1 = np.count_nonzero(water_interface)
        Area2 = np.count_nonzero(dt_inv==1)
        Area=np.mean((Area1, Area2))
        
        water_air_interface = interface_checker(pore_filling, water_interface, t, radius=1)
        Area_int1 = np.count_nonzero(water_air_interface)
        Area_int2 = np.count_nonzero(ndimage.distance_transform_cdt(water_air_interface)==1)
        Area_int = np.mean((Area_int1, Area_int2))
        Area_wet = Area - Area_int
        
#        interface[t,:,:,:][np.where(water_air_interface)] = True
        
        A_int[t] = Area_int
        A_wet[t] = Area_wet
        
    return A_int, A_wet
#samples = [samples[0]]   #for testing
data = {}
#pseudo: for sample  in samples
#sample = samples[2]
Energy={}
for sample in samples:
    if not sample[:3] == 'dyn': continue
    sample_data = xr.load_dataset(os.path.join(sourceFolder, sample))
    pore_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['pore_props_', sample[9:]])))
    
    vx = sample_data.attrs['voxel']
    
    name = sample_data.attrs['name']
    
    transitions = sample_data['transition_matrix']
    label_matrix = sample_data['label_matrix'].data
    time = sample_data['time'].data
    labels = sample_data['label'].data
    
    
    #E_kin = np.zeros([len(labels), len(time)])
    A_mean = pore_data['value_properties'].sel(property = 'mean_area').data
    
    E_kin = sample_data['volume'][:,1:]*vx/2/rho/A_mean[:,None]**2 * (np.diff(sample_data['volume'])*vx**3/np.diff(time))**2
    
    Energy[sample] = E_kin
    
    
#    A_int = np.zeros([len(labels), len(time)])
#    A_wet = np.zeros([len(labels), len(time)])
    

    
    #pseudo for pore in pores:
    #pore = 0
    #for label in labels:
    
    #    pore = pore +1   
        #    objects = skimage.measure.label(water_air_interface, connectivity=3)
        #    
        #    count_obj = np.uniqu
        
    result = Parallel(n_jobs=num_cores)(delayed(pore_interface_tracking)(label, label_matrix, transitions, time) for label in labels)
    
    result = np.array(result)
    A_int = result[:, 0, :]
    A_wet = result[:, 1, :]
    
    data[sample] = result
    df = helmholtz_energy(A_int, A_wet)
    
    plt.figure()
    plt.plot(time, A_int.sum(axis=0)/A_int.sum(axis=0).min()); plt.plot(time[1:,], E_kin.sum(axis=0)/E_kin.sum(axis=0).max())
    plt.title(sample)
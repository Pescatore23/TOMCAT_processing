# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:50:10 2019

@author: firo
"""

import xarray as xr
import os
import robpylib
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
from scipy import ndimage


num_cores = mp.cpu_count()

rho = 997 #kg/m3
vx = 2.75E-6 #m
gamma = 72.7E-3 #N/m
theta = 50 #° plasma treated PET

drive = '//152.88.86.87/data118'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1200_dry_seg_aniso_sep'

sourceFolder = os.path.join(data_path, processing_version)

samples = os.listdir(sourceFolder) 


#FIXME: Idea: additionally extract tesselated surface and use this area to calulated Helmholtz energy at large t --> maybe more exact than counting pixels
# do this for pixels labeled as water-air interface and water-solid (divide by 2?, <-- test behavior of surface creation)
# then decide if to do this at each time step
# smooth E(t) to get more stable derivations

def helmholtz_energy(A_int, A_wet, per_pore=False, gamma=gamma, theta=theta/180*np.pi, vx=vx):
    if per_pore:
        dF = gamma*(np.diff(A_int)*vx**2 - np.cos(theta)*np.diff(A_wet)*vx**2)  
    else:        
        dF = gamma*(np.diff(A_int, axis = 1) - np.cos(theta)*np.diff(A_wet, axis = 1))
    F = gamma*A_int*vx**2 - np.cos(theta)*A_wet*vx**2
    return dF, F

def kinetic_energy(volume, time, A_mean, vx=vx, rho=rho):
    E_kin = rho/4 * (volume[:,:-1] + volume[:,1:])*vx /A_mean[:,None]**2 * ((volume[:,1:] - volume[:,:-1])*vx**3)**2 /(time[1:]-time[:-1])**2
    return E_kin

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

def process_time_step(t, water):
    dt = ndimage.distance_transform_cdt(water)
    dt_inv = ndimage.distance_transform_cdt(~water)
    
    water_interface = dt==1
    Area1 = np.count_nonzero(water_interface)
    Area2 = np.count_nonzero(dt_inv==1)
    Area = (Area1+Area2)/2
    
    water_air_interface = interface_checker(water, water_interface, t, radius = 1)
    
    Area_int1 = np.count_nonzero(water_air_interface)
    Area_int2 = np.count_nonzero(ndimage.distance_transform_cdt(water_air_interface)==1)
    Area_int = (Area_int1+Area_int2)/2
    
    Area_wet = Area - Area_int
    
    return Area_int, Area_wet
    


def pore_interface_tracking(label, label_matrix, transitions, time):
    A_int = np.zeros(len(time))
    A_wet = np.zeros(len(time))
    mask = label_matrix==label
    
    bounding_box = ndimage.find_objects(mask)[0]
    transition = transitions[bounding_box].data
    mask = mask[bounding_box]
    

    pore_filling = transition*mask
    watermask = pore_filling > 0
    
    for t in range(1,len(time)):
        A_int[t] = A_int[t-1]
        A_wet[t] = A_wet[t-1]
        if (pore_filling==t).any():
            water = watermask*(pore_filling<t+1)
            A_i, A_w = process_time_step(t, water)
            A_int[t] = A_i
            A_wet[t] = A_w
    return A_int, A_wet
            
 

for sample in samples:
    if not sample[:3] == 'dyn': continue
    
    sample_data = xr.load_dataset(os.path.join(sourceFolder, sample))
    pore_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['pore_props_', sample[9:]])))
    
    name = sample_data.attrs['name']
    if not name in robpylib.TOMCAT.INFO.samples_to_repeat: continue
    print(name)
    filename = os.path.join(sourceFolder, ''.join(['energy_data_', name, '.nc']))
    
#    if os.path.exists(filename): continue
    
    transitions = sample_data['transition_matrix']
    volume = sample_data['volume'].data
    label_matrix = sample_data['label_matrix'].data
    time = sample_data['time'].data
#    step_size = time.copy()
#    step_size[1:] = np.diff(time)
    labels = sample_data['label'].data
    
    A_mean = pore_data['value_properties'].sel(property = 'mean_area').data
    
    result = Parallel(n_jobs=num_cores)(delayed(pore_interface_tracking)(label, label_matrix, transitions, time) for label in labels)
    result = np.array(result)
    
    A_int = result[:, 0, :]
    A_wet = result[:, 1, :]
    
    dF, F = helmholtz_energy(A_int, A_wet)
    Ek = kinetic_energy(volume, time, A_mean)
    dEk = np.zeros(Ek.shape)
    dEk[:,1:] = np.diff(Ek, axis = 1)
    
    energy_data = xr.Dataset({'kinetic_energy': (['label', 'time'], Ek),
                              'diff_kin_E': (['label', 'time'], dEk),
                              'Helmholtz_energy': (['label', 'time'], F[:,1:]),
                              'diff_F': (['label', 'time'], dF)},
                        coords = {'label': labels,
                                  'time': time[1:]})
    energy_data.attrs = sample_data.attrs
    energy_data['kinetic_energy'].attrs['units'] = 'J'
    energy_data['diff_kin_E'].attrs['units'] = 'J'
    energy_data['Helmholtz_energy'].attrs['units'] = 'J'
    energy_data['diff_F'].attrs['units'] = 'J'
    energy_data['time'].attrs['units'] = 's'
    
    energy_data.to_netcdf(filename)
    
    
    
    
    
#            FIXME: continue
            
            
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:16:07 2020

@author: firo
"""

import xarray as xr
import os
import socket
host = socket.gethostname()
from joblib import Parallel, delayed
from scipy import ndimage
import numpy as np
import trimesh
from skimage import measure
import robpylib
from skimage.morphology import cube

pc = False
#  Part 1:
if host == 'ddm05307':
    pc = True
    num_cores = 12

#  Part2:
if host == 'DDM04060':
    pc = True
    num_cores = 12
    
if host == 'DDM04672':
    pc = True
    num_cores = 12

if pc == False: print('host is '+host+' , make sure you run the script on the proper machine')


rho = 997 #kg/m3
vx = 2.75E-6 #m
gamma = 72.7E-3 #N/m
theta = 50 #° plasma treated PET

# smoothing parameters
k = 0.1
lamb  = 0.6037
iterations = 10

drive = '//152.88.86.87/data118'
# drive = r"NAS"
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1200_dry_seg_aniso_sep'
folder1 = os.path.join(drive, 'Robert_TOMCAT_3')

sourceFolder = os.path.join(data_path, processing_version)

samples = os.listdir(sourceFolder) 


def surface_smoothing(verts, faces, k=k, lamb=lamb, iterations = iterations):
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh = trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=k, iterations=iterations)
    verts = mesh.vertices
    faces = mesh.faces
    return verts, faces

def interface_extraction(wet, void):
   dry = np.bitwise_and(void, ~wet)
   wet = ndimage.binary_dilation(input = wet, structure = cube(3).astype(np.bool))
   dry = ndimage.binary_dilation(input = dry, structure = cube(3).astype(np.bool))
   
   interface = np.bitwise_and(wet, dry)
   interface = np.bitwise_and(interface, void)
   # interface = skimage.morphology.binary_erosion(interface, selem=ball(1).astype(np.bool))
   # interface = skimage.morphology.binary_dilation(interface, selem=ball(1).astype(np.bool))
   
   return interface

def water_interface_extraction(wet, nwet, void):
    wet = ndimage.binary_dilation(input = wet, structure = cube(3).astype(np.bool))
    nwet = ndimage.binary_dilation(input = nwet, structure = cube(3).astype(np.bool))
    interface = np.bitwise_and(wet, nwet)
    interface = np.bitwise_and(interface, void)                               
    return interface


# def interface_per_time_step(transitions, void, watermask, t):
def interface_per_time_step(wet, void, watermask, t):
    A_ws = 0
    A_wa = 0
    A_tot = 0
    if np.any(wet):
        try:
            verts, faces, _, _ = measure.marching_cubes_lewiner(wet)
            verts, faces = surface_smoothing(verts, faces)
            A_tot = measure.mesh_surface_area(verts, faces)
        except:
            print(str(t)+' A_tot_failed')
            
        try:
            interface = interface_extraction(wet, void)
            verts, faces, _, _ = measure.marching_cubes_lewiner(interface)
            verts, faces = surface_smoothing(verts, faces)
            A_wa = measure.mesh_surface_area(verts, faces)/2
        except:
            print(str(t)+' A_wa_failed')
            
        try:
            wet_interface = water_interface_extraction(wet, void, void)
            verts, faces, _, _ = measure.marching_cubes_lewiner(wet_interface)
            verts, faces = surface_smoothing(verts, faces)
            A_ws = measure.mesh_surface_area(verts, faces)/2
        except:
            print(str(t)+' A_ws_failed')

    return A_ws, A_wa, A_tot

for sample in samples:
    if not sample[:3] == 'dyn': continue
    print(sample)
    data = xr.load_dataset(os.path.join(sourceFolder, sample))
    name = data.attrs['name']
    filename = os.path.join(sourceFolder, ''.join(['total_energy_data_', name, '.nc']))
    # if os.path.exists(filename): continue
    if name == 'T3_025_1': continue
    print(name)
    fiberpath = os.path.join(folder1, name, '01a_weka_segmented_dry', 'classified')
    fibers, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiberpath, track=False)
    void = fibers==0
    fibers = None
    transition = data['transition_matrix'].data
    void = void[:,:,transition.shape[2]]
    time = data['time'].data
    watermask = transition>0
    result = Parallel(n_jobs=num_cores)(delayed(interface_per_time_step)(((transition<t+1)*watermask).astype(np.uint8), void, watermask, t) for t in range(1,time.shape[0]))
    result = np.array(result)
    
    area_data = xr.Dataset({'interfaces': (['time', 'area'], result)},
                            coords = {'area': ['A_ws', 'A_wa', 'A_tot'],
                                      'time': time[1:]},
                            attrs = {'A_ws': 'water-solid-interface area',
                                     'A_wa': 'water-air-interface area',
                                     'A_tot': 'total water interface for check-up'}
                           )
    area_data.to_netcdf(filename)
    

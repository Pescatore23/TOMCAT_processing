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

num_cores = 16
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

drive = r'\\152.88.86.87\data118'
if host=='xerus.local': 
    drive = r"NAS"
    num_cores = 7
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1200_dry_seg_aniso_sep'
folder1 = os.path.join(drive, 'Robert_TOMCAT_3')

sourceFolder = os.path.join(data_path, processing_version)

samples = os.listdir(sourceFolder) 

smooth_decision = 'yes'

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

def solid_interface_extraction(wet, void):
    # swap comments for v5
    # wet = ndimage.binary_dilation(input = wet, structure = cube(3).astype(np.bool))
    void = ndimage.binary_dilation(input = void, structure = cube(3).astype(np.bool))
    
    interface = np.bitwise_and(wet, void)
    # interface = np.bitwise_and(interface, void)                               
    return interface


# def interface_per_time_step(transitions, void, watermask, t):
def interface_per_time_step(wet, void, fibermesh, smooth_decision = smooth_decision):#, watermask, t):
    A_ws = 0
    A_wa = 0
    A_tot = 0
    A_wv = 0
    if np.any(wet):
        try:
            verts, faces, _, _ = measure.marching_cubes_lewiner(wet)
            if smooth_decision == 'yes':
                verts, faces = surface_smoothing(verts, faces)
            A_tot = measure.mesh_surface_area(verts, faces)
        except:
            A_tot = np.nan
            # print(str(t)+' A_tot_failed')
        A_wv = np.count_nonzero(wet[:,:,0])+np.count_nonzero(wet[:,:,-1])
        
        
        # comment out for v 4
        # try:
            # interface = interface_extraction(wet, void)
            # verts, faces, _, _ = measure.marching_cubes_lewiner(interface)
            # verts, faces = surface_smoothing(verts, faces)
            # A_wa = measure.mesh_surface_area(verts, faces)/2
            
            
        # except:
            # A_wa = -1
            # print(str(t)+' A_wa_failed')
            
        try:
            wet_interface = solid_interface_extraction(wet, void)
            
            # comment out for v4, uncomment for v5
            verts, faces, _, _ = measure.marching_cubes_lewiner(wet_interface)
            verts, faces = surface_smoothing(verts, faces)
            
            # A_ws = measure.mesh_surface_area(verts, faces)/2
            # A_wa = A_tot-A_ws-A_wv
            

            
            # uncomment for v 4
            # fverts = fibermesh.vertices
            # ffaces = fibermesh.faces
            
            # uncomment for v5
            fverts = verts
            ffaces = faces
            
            
            fvert_int = np.int16(fverts)
            wet_int_mask = wet_interface[fvert_int[:,0], fvert_int[:,1], fvert_int[:,2]]
            faces_mask = np.all(wet_int_mask[ffaces], axis = 1)
            wet_faces = ffaces[faces_mask]
            A_ws = measure.mesh_surface_area(fverts, wet_faces)
            
            # v7
            interface = interface_extraction(wet, void)
            int_mask = interface[fvert_int[:,0], fvert_int[:,1], fvert_int[:,2]]
            int_faces_mask = np.all(int_mask[ffaces], axis = 1)
            int_faces = ffaces[int_faces_mask]
            A_wa = measure.mesh_surface_area(fverts, int_faces)
            
            # A_wa = A_tot-A_ws-A_wv
        except:
            A_ws=np.nan
            A_wa=np.nan
            # print(str(t)+' A_ws_failed')

    return A_ws, A_wa, A_tot, A_wv

samples.sort()
for sample in samples:
    if not sample[:3] == 'dyn': continue
    print(sample)
    data = xr.load_dataset(os.path.join(sourceFolder, sample))
    name = data.attrs['name']
    filename = os.path.join(sourceFolder, ''.join(['total_energy_data_v7_', name, '.nc']))
    if os.path.exists(filename): continue
    if name == 'T3_025_1': continue
    print(name)
    fiberpath = os.path.join(folder1, name, '01a_weka_segmented_dry', 'classified')
    fibers, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiberpath, track=False)
    void = fibers==0
    transition = data['transition_matrix'].data
    fibers = fibers[:,:,:transition.shape[2]]
    fibermesh = False
    # verts, faces, _, _ = measure.marching_cubes_lewiner(fibers)
    # fibermesh = trimesh.Trimesh(vertices = verts, faces=faces)
    # print('fibermesh marched')
    # if smooth_decision == 'yes':
        # fibermesh = trimesh.smoothing.filter_taubin(fibermesh, lamb=lamb, nu=k, iterations=iterations)
        # print('fibermesh smoothed')
    void = void[:,:,:transition.shape[2]]
    time = data['time'].data
    watermask = transition>0
    result = Parallel(n_jobs=num_cores)(delayed(interface_per_time_step)((transition<t+1)*watermask, void, fibermesh) for t in range(1,time.shape[0]))
    result = np.array(result)
    
    area_data = xr.Dataset({'interfaces': (['time', 'area'], result)},
                            coords = {'area': ['A_ws', 'A_wa', 'A_tot', 'A_wv'],
                                      'time': time[1:]})
    area_data.attrs = data.attrs
    area_data.attrs['A_ws'] = 'water-solid-interface area'
    area_data.attrs['A_wa'] = 'water-air-interface area'
    area_data.attrs['A_tot'] = 'total water interface for check-up'
    area_data.attrs['A_wv'] = 'virtual interface at FOV boundary'
    area_data.attrs['smooth'] = smooth_decision
    area_data.to_netcdf(filename)

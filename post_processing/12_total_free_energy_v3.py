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

num_cores = 8
temp_folder = None
pc = False


drive = r'\\152.88.86.87\data118'
drive = r'\\152.88.86.68\data118'
if host=='xerus.local': 
    drive = r"NAS"
    num_cores = 7
   
#  Part 1:
if host == 'ddm05307':
    drive = r'Z:'
    drive = r'Y:'
    temp_folder = r"F:\joblib_temp"
    pc = True
    num_cores = 5

#  Part2:
if host == 'DDM04060':
    pc = True
    num_cores = 12
    
if host == 'DDM04672':
    pc = True
    num_cores = 12
    
if host == 'DDM06609':
    pc = True
    num_cores = 8
    temp_folder = r"Z:\users\firo\joblib_tmp"

if host == 'hades':
    drive = '/home/firo/NAS'
    # drive = '/home/firo/NAS2'
    pc = True
    num_cores = 4
    
if host == 'mpc2053.psi.ch' or host=='mpc1833.psi.ch':
    pc = True
    num_cores = 8
    temp_folder = "/mnt/nas_Uwrite/fische_r/joblib_temp"

if pc == False: print('host is '+host+' , make sure you run the script on the proper machine')


rho = 997 #kg/m3
vx = 2.75E-6 #m
gamma = 72.7E-3 #N/m
theta = 50 #° plasma treated PET

# smoothing parameters
k = 0.1
lamb  = 0.6037
iterations = 10


data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1200_dry_seg_aniso_sep'
# folder1 = os.path.join(drive, 'Robert_TOMCAT_3')
# folder1 = os.path.join(drive, 'Robert_TOMCAT_4')
folder1 = "/mnt/nas_Uwrite/fische_r/NAS_data"
# folder1 = os.path.join(r"Y:", 'Robert_TOMCAT_5_split')
# sourceFolder = os.path.join(data_path, processing_version)
# sourceFolder = os.path.join(drive, 'Robert_TOMCAT_5_netcdf4')
# sourceFolder = r"A:\Robert_TOMCAT_4_netcdf4_split_v2_no_pore_size_lim"
sourceFolder = "/mnt/nas_Uwrite/fische_r/NAS_data"
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
   # wet = ndimage.binary_dilation(input = wet, structure = cube(3).astype(np.bool))
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
def interface_per_time_step(wet, void, smooth_decision = smooth_decision):#, watermask, t):
    A_ws = 0
    A_wa = 0
    A_tot = 0
    A_wv = 0
    A_ws_label = 0
    A_wa_corr = 0
    A_wa_corr2 = 0
    if np.any(wet):
        try:
            verts, faces, _, _ = measure.marching_cubes_lewiner(wet)
            if smooth_decision == 'yes':
                verts, faces = surface_smoothing(verts, faces)
            A_tot = measure.mesh_surface_area(verts, faces)
            a = True
        except:
            A_tot = np.nan
            a = False
            # print(str(t)+' A_tot_failed')
        A_wv = np.count_nonzero(wet[:,:,0])+np.count_nonzero(wet[:,:,-1])
        
        if a:
            # wet surface
            try:
                wet_interface = solid_interface_extraction(wet, ~void)
                sverts, sfaces, _, _ = measure.marching_cubes_lewiner(wet_interface)
                if smooth_decision == 'yes':
                    sverts, sfaces = surface_smoothing(sverts, sfaces)  
                A_ws = measure.mesh_surface_area(sverts, sfaces)/2           
                
                fverts = verts
                ffaces = faces
                fvert_int = np.int16(fverts)
                wet_mask = wet_interface[fvert_int[:,0], fvert_int[:,1], fvert_int[:,2]]
                ffaces_mask = np.all(wet_mask[ffaces], axis = 1)
                wffaces = ffaces[ffaces_mask]
                A_ws_label = measure.mesh_surface_area(fverts, wffaces)
            except:
                A_ws = 0
                A_ws_label = 0
            
            A_wa_corr = A_tot - A_ws - A_wv
            A_wa_corr2 = A_tot - A_ws_label - A_wv
                # print(str(t)+' A_ws_failed')

    return A_ws, A_wa, A_tot, A_wv, A_ws_label, A_wa_corr, A_wa_corr2

samples.sort()
for sample in samples:
    if not sample[:3] == 'dyn': continue
    print(sample)
    data = xr.open_dataset(os.path.join(sourceFolder, sample))
    name = data.attrs['name']
    filename = os.path.join(sourceFolder, ''.join(['total_energy_data_v3_1_', name, '.nc']))
    # filename = os.path.join( r'R:\Scratch\305\_Robert',''.join(['total_energy_data_v3_1_', name, '.nc']))
    if os.path.exists(filename):
        print('already done')
        continue
    if name == 'T3_025_1': continue
    print(name)
    fiberpath = os.path.join(folder1, name, '01a_weka_segmented_dry', 'classified')
    time = data['time'].data
    data.close()
    fibers, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiberpath, track=False)
    void = fibers==0
    transition = data['transition_matrix'].data
    void = void[:,:,:transition.shape[2]]
    watermask = transition>0
    result = Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(interface_per_time_step)((transition<t+1)*watermask, void) for t in range(1,time.shape[0]))
    result = np.array(result)
    
    area_data = xr.Dataset({'interfaces': (['time', 'area'], result)},
                            coords = {'area': ['A_ws', 'A_wa', 'A_tot', 'A_wv', 'A_ws_label', 'A_wa_corr', 'A_wa_corr2'],
                                      'time': time[1:]})
    area_data.attrs = data.attrs
    area_data.attrs['A_ws'] = 'water-solid-interface area'
    area_data.attrs['A_wa'] = 'water-air-interface area'
    area_data.attrs['A_tot'] = 'total water interface for check-up'
    area_data.attrs['A_wv'] = 'virtual interface at FOV boundary'
    area_data.attrs['A_ws_label'] = 'wet solid area by labeling watermesh'
    area_data.attrs['A_wa_corr'] = 'water-air-interface as difference of total waster surface and wet solid'
    area_data.attrs['A_wa_corr2'] = 'water-air-interface as difference of total waster surface and wet solid (by labeling)'
    area_data.attrs['smooth'] = smooth_decision
    area_data.to_netcdf(filename)

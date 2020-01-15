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
# import skimage.morphology
from skimage.morphology import cube
from skimage import measure
import trimesh

num_cores = mp.cpu_count()

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
folder2 = os.path.join(drive, 'Robert_TOMCAT_3_part_2')

sourceFolder = os.path.join(data_path, processing_version)

samples = os.listdir(sourceFolder) 

def surface_smoothing(verts, faces, k=k, lamb=lamb, iterations = iterations):
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh = trimesh.smoothing.filter_taubin(mesh, lamb=lamb, nu=k, iterations=iterations)
    verts = mesh.vertices
    faces = mesh.faces
    return verts, faces

def helmholtz_energy(A_int, A_wet, per_pore=False, gamma=gamma, theta=theta/180*np.pi, vx=vx):
    if per_pore:
        dF = gamma*(np.diff(A_int)*vx**2 - np.cos(theta)*np.diff(A_wet)*vx**2)  
    else:        
        dF = gamma*(np.diff(A_int, axis = 1)*vx**2 - np.cos(theta)*np.diff(A_wet, axis = 1)*vx**2)
    F = gamma*A_int*vx**2 - np.cos(theta)*A_wet*vx**2
    return dF, F

def kinetic_energy(volume, time, A_mean, vx=vx, rho=rho):
    E_kin = rho/4 * (volume[:,:-1] + volume[:,1:])*vx /A_mean[:,None]**2 * ((volume[:,1:] - volume[:,:-1])*vx**3)**2 /(time[1:]-time[:-1])**2
    return E_kin

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
                               
def measure_interfaces(label, label_matrix, transition, void, time):
    A_int = np.zeros(len(time))
    A_wet = np.zeros(len(time))
    A_nw = np.zeros(len(time))
    label_obj = label_matrix==label
    watermask = transition > 0
    pore_filling = transition*label_obj
    neighbor_water = transition*(~label_obj)
    

    for t in range(1,len(time)):
        A_int[t] = A_int[t-1]
        A_wet[t] = A_wet[t-1]
        A_nw[t] = A_nw[t-1]
        if (pore_filling == t).any():
            wet = watermask * (pore_filling < t+1)
            n_wet = watermask * (neighbor_water < t+1)
            # wet = skimage.morphology.binary_erosion(wet, selem=ball(1).astype(np.bool))
            # wet = skimage.morphology.binary_dilation(wet, selem=ball(1).astype(np.bool))
            
            wet_interface = water_interface_extraction(wet, void, void)
            
            try:
                # verts, faces, _, _ = measure.marching_cubes_lewiner(wet)
                # verts, faces = surface_smoothing(verts, faces)
                # A = measure.mesh_surface_area(verts, faces)
                
                # for version9:
                verts, faces, _, _ = measure.marching_cubes_lewiner(wet_interface)
                verts, faces = surface_smoothing(verts, faces)
                Aw = measure.mesh_surface_area(verts, faces)/2
            except:
                # A = 0 # commented out for version 9
                Aw = 0
            # try:
            interface = interface_extraction(wet, void)
            w_w_inter = water_interface_extraction(wet, n_wet, void)
            interface[w_w_inter] = False
            
            try:
                verts, faces, _, _ = measure.marching_cubes_lewiner(interface)
                verts, faces = surface_smoothing(verts, faces)
                A_wa = measure.mesh_surface_area(verts, faces)/2
            except:
                A_wa = 0
            
            try:
                verts, faces, _, _ = measure.marching_cubes_lewiner(w_w_inter)
                verts, faces = surface_smoothing(verts, faces)
                A_ww = measure.mesh_surface_area(verts, faces)/2
            except:
                A_ww = 0
            # Aw = A-A_wa-A_ww   # commented out for version 9
            if Aw < 0: Aw=0
            A_wet[t] = Aw
            A_int[t] = A_wa
            A_nw[t] = A_ww
            

    return A_int, A_wet, A_nw
            
 

for sample in samples:
    if not sample[:3] == 'dyn': continue
    
    sample_data = xr.load_dataset(os.path.join(sourceFolder, sample))
    pore_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['pore_props_', sample[9:]])))
    
    
    # FIXME load fiber images to get real void geometry
    
    name = sample_data.attrs['name']
    
    if name in os.listdir(folder1):
        fiberpath = os.path.join(folder1, name, '01a_weka_segmented_dry', 'classified')
    
    if name in os.listdir(folder2):
        fiberpath = os.path.join(folder2, name, '01a_weka_segmented_dry', 'classified')
        
    fibers, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiberpath, track=False)
    
    void = fibers==0
    fibers = None
    
    # if name in robpylib.TOMCAT.INFO.samples_to_repeat: continue
    print(name)
    filename = os.path.join(sourceFolder, ''.join(['energy_data_v9_', name, '.nc']))
    
    # if os.path.exists(filename): continue
    
    transitions = sample_data['transition_matrix'].data
    volume = sample_data['volume'].data
    label_matrix = sample_data['label_matrix'].data
    time = sample_data['time'].data
#    step_size = time.copy()
#    step_size[1:] = np.diff(time)
    labels = sample_data['label'].data
    
    A_mean = pore_data['value_properties'].sel(property = 'mean_area').data
    
    crude_labels = np.unique(label_matrix)[1:]
    crude_pores = ndimage.find_objects(label_matrix)
    
    pores = []
    for (pore, label) in zip(crude_pores, crude_labels):
        if label in labels:
            pores.append(pore)
    
    crude_pores = None
    
    bounding_boxes = []
    shape = label_matrix.shape
    for pore in pores:
        bounding_boxes.append(robpylib.CommonFunctions.pore_network.extend_bounding_box(pore, shape))
    
    # [bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop]
    result = Parallel(n_jobs=num_cores)(delayed(measure_interfaces)(label, label_matrix[bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop], transitions[bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop], void[bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop], time) for (label, bb) in zip(labels, bounding_boxes))
    result = np.array(result)
    
    A_int = result[:, 0, :]
    A_wet = result[:, 1, :]
    A_nw = result[:, 2, :]
    
    dF, F = helmholtz_energy(A_int, A_wet)
    Ek = kinetic_energy(volume, time, A_mean)
    dEk = np.zeros(Ek.shape)
    dEk[:,1:] = np.diff(Ek, axis = 1)
    
    energy_data = xr.Dataset({'kinetic_energy': (['label', 'time'], Ek),
                              'diff_kin_E': (['label', 'time'], dEk),
                              'Helmholtz_energy': (['label', 'time2'], F),
                              'diff_F': (['label', 'time'], dF),
                              'interface_area': (['label', 'time2'], A_int),
                              'wet_area': (['label', 'time2'], A_wet),
                              'water_water_interface': (['label', 'time2'], A_nw),
                              'smoothing': ('parameter', np.array([k, lamb, iterations]))},
                        coords = {'label': labels,
                                  'time': time[1:],
                                  'time2': time,
                                  'parameter': ['k', 'lambda', 'iterations']},
                        attrs = {'comment': 'surfaces measured as triangulated mesh instead of counting pixels + noise removal'})
    
    energy_data.attrs = sample_data.attrs
    energy_data['kinetic_energy'].attrs['units'] = 'J'
    energy_data['diff_kin_E'].attrs['units'] = 'J'
    energy_data['Helmholtz_energy'].attrs['units'] = 'J'
    energy_data['diff_F'].attrs['units'] = 'J'
    energy_data['time'].attrs['units'] = 's'
    
    energy_data.to_netcdf(filename)
    
    
    
    
    
#            FIXME: continue
            
            
    
    
    
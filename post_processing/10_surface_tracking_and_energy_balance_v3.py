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
# import joblib
# import multiprocessing as mp
from scipy import ndimage
# import skimage.morphology
from skimage.morphology import cube
from skimage import measure
import trimesh
# from dask.distributed import Client
# client = Client(processes=False)             # create local cluster

num_cores = 30#mp.cpu_count()

rho = 997 #kg/m3
vx = 2.75E-6 #m
gamma = 72.7E-3 #N/m
theta = 50 #° plasma treated PET

# smoothing parameters
k = 0.1
lamb  = 0.6037
iterations = 10
smooth_decision = 'yes'

drive = r'\\152.88.86.87\data118'
# drive = r"NAS"
# drive =  r'Z:\'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
# data_path = r'Z:\Robert_TOMCAT_3_netcdf4_archives'
processing_version = 'processed_1200_dry_seg_aniso_sep'

folder = os.path.join(drive, 'Robert_TOMCAT_3')
# folder = r'Z:\Robert_TOMCAT_3'

sourceFolder = os.path.join(data_path, processing_version)
# sourceFolder = os.path.join(drive,  'Robert_TOMCAT_3_netcdf4_archives', processing_version)

samples = os.listdir(sourceFolder) 


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

def water_interface_extraction(wet, nwet, void):
    # wet = ndimage.binary_dilation(input = wet, structure = cube(3).astype(np.bool))
    nwet = ndimage.binary_dilation(input = nwet, structure = cube(3).astype(np.bool))
    interface = np.bitwise_and(wet, nwet)
    interface = np.bitwise_and(interface, void)                               
    return interface

def water_interface_extraction2(wet, nwet, void):
    wet = ndimage.binary_dilation(input = wet, structure = cube(3).astype(np.bool))
    nwet = ndimage.binary_dilation(input = nwet, structure = cube(3).astype(np.bool))
    interface = np.bitwise_and(wet, nwet)
    interface = np.bitwise_and(interface, void)                               
    return interface


def solid_interface_extraction(wet, solid):
    # swap dilation for v13b
    # wet = ndimage.binary_dilation(input = wet, structure = cube(3).astype(np.bool))
    solid = ndimage.binary_dilation(input = solid, structure = cube(3).astype(np.bool))
    interface = np.bitwise_and(wet, solid)
    # interface = np.bitwise_and(interface, void)                               
    return interface
                               
def measure_interfaces(label, label_matrix, transition, void, time, bb, smooth_decision = smooth_decision):
    A_wa = np.zeros(len(time))
    A_wa_corr = np.zeros(len(time))
    A_ws = np.zeros(len(time))
    A_ws_label = np.zeros(len(time))
    A_ww_label = np.zeros(len(time))
    A_ww = np.zeros(len(time))
    A_tot = np.zeros(len(time))
    label_obj = label_matrix==label
    watermask = transition > 0
    pore_filling = transition*label_obj
    pore_watermask = watermask*label_obj
    neighbor_water = transition*(~label_obj)
    # x0 = bb[0].start
    # y0 = bb[1].start
    # z0 = bb[2].start
    for t in range(1,len(time)):
        A_wa[t] = A_wa[t-1]
        A_wa_corr[t] = A_wa_corr[t-1]
        A_ws[t] = A_ws[t-1]
        A_ws_label[t] = A_ws_label[t-1]
        A_ww_label[t] = A_ww_label[t-1]
        A_ww[t] = A_ww[t-1]
        A_tot[t] = A_tot[t-1]
        if (pore_filling == t).any():
            wet = pore_watermask * (pore_filling < t+1)
            nwet = watermask * (neighbor_water < t+1)
            # wet_interface = solid_interface_extraction(wet, void)
            
            # total water surface
            try: 
                wverts, wfaces, _, _ = measure.marching_cubes_lewiner(wet)
                if smooth_decision == 'yes':
                    wverts, wfaces = surface_smoothing(wverts, wfaces)
                Atot = measure.mesh_surface_area(wverts, wfaces)
                a = True
            except:
                a = False
            
            if a:
            # water interface
                try:
                    interface = interface_extraction(wet, void)
                    iverts, ifaces, _, _ = measure.marching_cubes_lewiner(interface)
                    if smooth_decision == 'yes':
                        iverts, ifaces = surface_smoothing(iverts, ifaces)
                    # wvert_int = np.int16(wverts)
                    # int_mask = interface[wvert_int[:,0], wvert_int[:,1], wvert_int[:,2]]
                    # ifaces_mask = np.all(int_mask[wfaces], axis=1)
                    # ifaces = wfaces[ifaces_mask]
                    Awa = measure.mesh_surface_area(iverts, ifaces)/2
                    d = True
            # virtual interface at pore boundary
                except:
                    d = False
                    Awa = 0
                    
                try:
                    virtual_interface = water_interface_extraction(wet, nwet, void)
                    vverts, vfaces, _, _ = measure.marching_cubes_lewiner(virtual_interface)
                    if smooth_decision == 'yes':
                        vverts, vfaces = surface_smoothing(vverts, vfaces)
                    # wvert_int = np.int16(wverts)
                    # virtual_mask = virtual_interface[wvert_int[:,0], wvert_int[:,1], wvert_int[:,2]]
                    # vfaces_mask = np.all(virtual_mask[wfaces], axis=1)
                    # vfaces = wfaces[vfaces_mask]
                    Aww = measure.mesh_surface_area(vverts, vfaces)/2
                    
                    # virtual_interface = water_interface_extraction(nwet, wet, void)
                    virtual_interface = water_interface_extraction2(wet, nwet, void)
                    tverts = wverts
                    tfaces = wfaces
                    tvert_int = np.in16(tverts)
                    virtual_mask = virtual_interface[tvert_int[:,0], tvert_int[:,1], tvert_int[:,2]]
                    tfaces_mask = np.all(virtual_mask[tfaces], axis =1)
                    vtfaces = tfaces[tfaces_mask]
                    Aww_label = measure.mesh_surface_area(tverts, vtfaces)
                    
                    
                    b = True
                except:
                    b = False
                    Aww = 0
                    Aww_label = 0
                    
            # wet surface
                try:
                    wet_interface = solid_interface_extraction(wet, ~void)
                    sverts, sfaces, _, _ = measure.marching_cubes_lewiner(wet_interface)
                    if smooth_decision == 'yes':
                        sverts, sfaces = surface_smoothing(sverts, sfaces)  
                    Aws = measure.mesh_surface_area(sverts, sfaces)/2           
                    
                    fverts = wverts
                    ffaces = wfaces
                    fvert_int = np.int16(fverts)
                    wet_mask = wet_interface[fvert_int[:,0], fvert_int[:,1], fvert_int[:,2]]
                    ffaces_mask = np.all(wet_mask[ffaces], axis = 1)
                    wffaces = ffaces[ffaces_mask]
                    Aws_label = measure.mesh_surface_area(fverts, wffaces)
                    c = True
                except:
                    c = False
                    Aws = 0
                    Aws_label = 0
                    
                A_tot[t] = Atot
                if b:
                    A_ww[t] = Aww
                    A_ww_label[t] = Aww_label
                if c:
                    A_ws[t] = Aws
                    A_ws_label[t] = Aws_label
                    A_wa_corr[t] = Atot-Aws-Aww
                # A_wa[t] = Atot-Aww-Aws
                if d:
                    A_wa[t] = Awa
                
            # old version (v11)
            # # try:
            # #     verts, faces, _, _ = measure.marching_cubes_lewiner(wet_interface)
            # #     verts, faces = surface_smoothing(verts, faces)
            # #     Aws = measure.mesh_surface_area(verts, faces)/2
            # #     a=True
            # # except:
            # #     a=False

            # # interface = interface_extraction(wet, void)
            # w_w_inter = water_interface_extraction(wet, n_wet, void)
            # # interface[w_w_inter] = False
            
            # # try:
            # #     verts, faces, _, _ = measure.marching_cubes_lewiner(interface)
            # #     verts, faces = surface_smoothing(verts, faces)
            # #     Awa = measure.mesh_surface_area(verts, faces)/2
            # #     b=True
            # # except:
            # #     b=False
            
            # try:
            #     verts, faces, _, _ = measure.marching_cubes_lewiner(wet)
            #     verts, faces = surface_smoothing(verts, faces)
            #     Atot = measure.mesh_surface_area(verts, faces)
            #     b=True
            # except:
            #     b=False
            
            # try:
            #     verts, faces, _, _ = measure.marching_cubes_lewiner(w_w_inter)
            #     verts, faces = surface_smoothing(verts, faces)
            #     Aww = measure.mesh_surface_area(verts, faces)/2
            #     c=True
            # except:
            #     c=False
                
            # if a: A_ws[t] = Aws
            # if b:
            #     if Aws<=Atot: A_wa[t] = Atot-Aws
            # if b: A_tot[t] = Atot
            # if c: A_ww[t] = Aww
        
    return A_wa, A_ws, A_ww, A_tot, A_wa_corr, A_ws_label, A_ww_label
    # A_wa = result[:, 0, :]
    # A_ws = result[:, 1, :]
    # A_ww = result[:, 2, :]
    # A_tot = result[:, 3, :]     up to v16, Awa and Aws mixed up!!!
 
samples.sort()
for sample in samples:
    if not sample[:3] == 'dyn': continue
    
    sample_data = xr.load_dataset(os.path.join(sourceFolder, sample))
    # pore_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['pore_props_', sample[9:]])))
    
    
    # FIXME load fiber images to get real void geometry
    
    name = sample_data.attrs['name']
    if name == 'T3_025_1': continue
    print(name)
    # fiberpath = os.path.join(folder, name, '01a_weka_segmented_dry', 'classified')
            
    # fibers, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiberpath, track=False)
    fiber_path = os.path.join(data_path, 'fiber_data', ''.join(['fiber_data_',name,'.nc']))
    fiber_data = xr.load_dataset(fiber_path)

    fibers = fiber_data['fibers'].data
    
    transitions = sample_data['transition_matrix'].data
    fibers = fibers[:,:,:transitions.shape[2]]

    
    void = fibers==0
    # verts, faces, _, _ = measure.marching_cubes_lewiner(fibers)
    fibers = fibers >0
    fibermesh = False
    # print('fibermesh marched')
    # fibermesh = trimesh.Trimesh(vertices = verts, faces=faces)
    # if smooth_decision == 'yes':
        # fibermesh = trimesh.smoothing.filter_taubin(fibermesh, lamb=lamb, nu=k, iterations=iterations)
        # print('fibermesh smoothed')
    # if name in robpylib.TOMCAT.INFO.samples_to_repeat: continue
    
    filename = os.path.join(sourceFolder, ''.join(['energy_data_v3_5_', name, '.nc']))
    
    if os.path.exists(filename): continue
    
    transitions = sample_data['transition_matrix'].data
    volume = sample_data['volume'].data
    label_matrix = sample_data['label_matrix'].data
    time = sample_data['time'].data
#    step_size = time.copy()
#    step_size[1:] = np.diff(time)
    labels = sample_data['label'].data
    
    # A_mean = pore_data['value_properties'].sel(property = 'mean_area').data
    
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
    print('start parallel computing')
    # [bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop]
    # with joblib.parallel_backend('dask'):
    result = Parallel(n_jobs=num_cores)(delayed(measure_interfaces)(label, label_matrix[bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop], transitions[bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop], void[bb[0].start:bb[0].stop, bb[1].start:bb[1].stop, bb[2].start:bb[2].stop], time, bb) for (label, bb) in zip(labels, bounding_boxes))
    result = np.array(result)
    
   # return A_wa, A_ws, A_ww, A_tot, A_wa_corr, A_ws_label
    A_wa = result[:, 0, :]
    A_ws = result[:, 1, :]
    A_ww = result[:, 2, :]
    A_tot = result[:, 3, :]
    A_wa_corr = result[:, 4, :]
    A_ws_label = result[:, 5, :]
    A_ww_label = result[:, 6, :]
    

    
    energy_data = xr.Dataset({'water_air_area': (['label', 'time'], A_wa),
                              'water_solid_area': (['label', 'time'], A_ws),
                              'water_water_area': (['label', 'time'], A_ww),
                              'total_water_surface': (['label', 'time'], A_tot),
                              'water_solid_area_by_label': (['label', 'time'], A_ws_label),
                              'water_water_area_by_label': (['label', 'time'], A_ww_label),
                              'water_air_area_by_difference': (['label', 'time'], A_wa_corr),
                              'smoothing': ('parameter', np.array([k, lamb, iterations]))},
                        coords = {'label': labels,
                                  'time': time,
                                  'parameter': ['k', 'lambda', 'iterations']},
                        attrs = {'comment': 'surfaces measured as triangulated mesh instead of counting pixels + noise removal',
                                  'pixel_size': '2.75um',
                                  'px': 2.75E-6})
    
    energy_data.attrs = sample_data.attrs
    # energy_data['kinetic_energy'].attrs['units'] = 'J'
    # energy_data['diff_kin_E'].attrs['units'] = 'J'
    # energy_data['Helmholtz_energy'].attrs['units'] = 'J'
    # energy_data['diff_F'].attrs['units'] = 'J'
    energy_data['time'].attrs['units'] = 's'
    energy_data['water_air_area'].attrs['units'] = 'px'
    energy_data['water_solid_area'].attrs['units'] = 'px'
    energy_data['water_water_area'].attrs['units'] = 'px'
    energy_data['water_water_area_by_label'].attrs['units'] = 'px'
    energy_data['water_solid_area_by_label'].attrs['units'] = 'px'
    energy_data['total_water_surface'].attrs['units'] = 'px'
    energy_data['water_air_area_by_difference'].attrs['units'] = 'px'
    energy_data.attrs['smoothed'] = smooth_decision
    
    
    energy_data.to_netcdf(filename)
    
    
    
    
    
#            FIXME: continue
            
            
    
    
    
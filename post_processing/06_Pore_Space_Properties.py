# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:36:15 2019

@author: firo
"""

import numpy as np
from scipy import ndimage
from skimage import measure
import xarray as xr
import os
#import time
from joblib import Parallel, delayed
import h5py
# import multiprocessing as mp


num_cores = 16#mp.cpu_count()
# drive = '//152.88.86.87/data118'
drive = r"A:"
# processing_version = 'processed_1200_dry_seg_aniso_sep'
# data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives', processing_version)
data_path = os.path.join(drive, 'Robert_TOMCAT_4_netcdf4_split_v2')
# data_path = r"Z:\Robert_TOMCAT_3_combined_archives"
# data_path = os.path.join(drive, 'Robert_TOMCAT_3b_netcdf4')
# data_path = r"Z:\Robert_TOMCAT_3_netcdf4_archives\expandedlabels"
# data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives', 'expandedlabels')

def cylinder_coords(x, y, x0=0, y0=0):
    r = np.sqrt( (x-x0)**2 + (y-y0)**2)
    phi = np.arctan2(y-y0, x-x0)
    return r, phi


def reduced_pore_object(labels, label):
    pore = labels == label
    
    # pore = np.uint8(pore)

#    time_0 = time.time()w
    
#    print('BB')
    bounding_box = ndimage.find_objects(pore)[0]
#    time_BB = time.time()
#    print(time_BB-time_0) 

    pore_object = pore[bounding_box].copy()
    return pore_object, bounding_box, label

def get_pore_props(pore_object, bounding_box, label):

    COM = ndimage.measurements.center_of_mass(pore_object)
    COM = COM + np.array([bounding_box[0].start, bounding_box[1].start, bounding_box[2].start])         
           
    inertia_tensor = measure._moments.inertia_tensor(pore_object)
    in_tens_eig = np.linalg.eigvalsh(inertia_tensor)
    volume = np.array(np.count_nonzero(pore_object))
    
    major_axis = 4*np.sqrt(in_tens_eig[-1])
    minor_axis = 4*np.sqrt(in_tens_eig[0])
    
    aspect_ratio = major_axis/minor_axis

    z_extent = pore_object.shape[2]
    areas_2D = np.zeros(z_extent)
    perim_2D = areas_2D.copy()
    COM_2D = np.zeros([z_extent, 2])
    
    
#    FIXME: perimeter calculation might be flawed, shape factor of circle is 0.08
#    regionprops erodes image to get the interface pixels --> underpredicts perimeter
#    test: binary dilate cross_section, possible problem: reaches boundary box
    for z in range(z_extent):
        cross_section = pore_object[:,:,z]
        areas_2D[z] = np.count_nonzero(cross_section)
        perim_2D[z] = measure._regionprops.perimeter(ndimage.binary_dilation(cross_section))
        COM_2D[z, :] = ndimage.measurements.center_of_mass(cross_section)
        
    shape_factor_2D = areas_2D/perim_2D**2
    shape_factor_2D = shape_factor_2D[np.where(perim_2D>0)]
    
    median_shape_factor = np.median(shape_factor_2D)
    mean_shape_factor = np.mean(shape_factor_2D)
    shape_factor_std = shape_factor_2D.std()
    
    mean_area = np.mean(areas_2D)
    median_area = np.median(areas_2D)
    area_std = areas_2D.std()

    vert_axis = COM_2D.mean(axis=0)
    
    x = COM_2D[:,0]
    y = COM_2D[:,1]
    z = np.arange(z_extent)
    
    try:
        x_fit = np.polyfit(z, x, 1)
        y_fit = np.polyfit(z, y, 1)
        fit_state = np.array(1)
    except:
        x_fit = [0, 0]
        y_fit = [0, 0]
        fit_state =  np.array(0)
    
    x_fun = np.poly1d(x_fit)
    y_fun = np.poly1d(y_fit)
    
    tilt_axis = np.array([x_fit[0], y_fit[0]])  #slopes of x(z) and y(z) # ist das nicht die major axis im inertia_tensor? ..., nein!, weil die Pore nicht immer gleich doick ist
    
    rho_v, phi_v = cylinder_coords(COM_2D[:,0], COM_2D[:,1], vert_axis[0], vert_axis[1])
    rho_t, phi_t = cylinder_coords(COM_2D[:,0], COM_2D[:,1], x_fun(z), y_fun(z))
    
    
    v_ecc = rho_v.mean()
    shell_mob = rho_v.std()/v_ecc     # radius std could be interesting measure for inter-yarn-layer mobility, but get relative 
#    t_ecc = rho_t.std()
    
    t_ecc = rho_t.mean()
    
    dx = np.diff(x)
    dy = np.diff(y)
    
    delta_phi = np.abs(np.diff(phi_v)).sum()
    
    dir_length = np.sqrt( (x[0]-x[-1])**2 + (y[0]-y[-1])**2+ z_extent**2)
    arc_length = np.sqrt(dx**2 + dy**2 + 1).sum()
    tort = arc_length/dir_length
        
    results = {'center_of_mass': COM,
               'bounding_box': bounding_box,
               'inertia_tensor': inertia_tensor,
               'inertia_tensor_eigs': in_tens_eig,
               'major_axis': major_axis,
               'minor_axis': minor_axis,
               'aspect_ratio': aspect_ratio,
               'median_shape_factor': median_shape_factor,
               'mean_shape_factor': mean_shape_factor,
               'shape_factor_std': shape_factor_std,
               'shp_fac_rel_var': shape_factor_std/mean_shape_factor,
               'tilt_axis_state': fit_state,
               'tilt_axis': tilt_axis,
               'tilt_angle_grad': np.arctan(np.mean([np.abs(tilt_axis)]))*180,
               'eccentricity_from_vertical_axis': v_ecc,
               'eccentricity_from_tilt_axis': t_ecc,
               'distance_end_to_end': dir_length,
               'arc_length': arc_length,
               'tortuosity': tort,
               'volume': volume,
               'mean_area': mean_area,
               'median_area': median_area,
               'area_std': area_std,
               'area_rel_var': area_std/mean_area,
               'angular_dist': delta_phi,
               'shell_mobility': shell_mob
            }
    
    return label, results
    
#c=0
liste = os.listdir(data_path)
# liste.reverse()
for filename in liste:
    if not filename[:3] == 'dyn': continue
    file = os.path.join(data_path, filename)
#    if c>0: continue
    datafile = h5py.File(file)
    name = datafile.attrs['name'].decode()
    datafile.close()
    
    new_filename = ''.join(['pore_props_', name,'.nc']) #'_size_',str(dyn_data.attrs['size_factor'])
    print(name)
    # if dyn_data.attrs['name'] == 'T4_300_4_III': continue
    # if dyn_data.attrs['name'] == 'T4_025_4': continue
    if os.path.exists(os.path.join(data_path, new_filename)): continue
    dyn_data = xr.load_dataset(file)
    label_matrix = dyn_data['label_matrix'].data
    labels = dyn_data['label'].data
    
    # for artificial data only:
    # labels = np.unique(label_matrix)[1:]
    
    pore_objects = []
    for label in labels:
        pore_objects.append(reduced_pore_object(label_matrix, label))
    
    pore_props = Parallel(n_jobs=num_cores)(delayed(get_pore_props)(*pore_object) for pore_object in pore_objects)
    
    properties_val = []
    properties_vect = []
    max_vec_dim = 0
    props = list(pore_props[0][1].keys())
    for prop in props:
        if prop == 'bounding_box': continue
        if len(pore_props[0][1][prop].shape) < 1:
            properties_val.append(prop)
        if len(pore_props[0][1][prop].shape) == 1:
            properties_vect.append(prop)
            max_vec_dim = max(max_vec_dim, len(pore_props[0][1][prop]))
    
    val_props = np.zeros([len(labels), len(properties_val)])
    vec_props = np.zeros([len(labels), len(properties_vect), max_vec_dim])
    
    l = 0
    for label in labels:
        p = 0
        pv = 0
        for prop in properties_val:
            val_props[l,p] = pore_props[l][1][prop]
            p = p+1
        for prop_v in properties_vect:
            data = pore_props[l][1][prop_v]
            vec_props[l,pv,:len(data)] = data
            pv =pv+1
        l=l+1
    
    val_props[np.isnan(val_props)] = 0
    val_props[np.isinf(val_props)] = 0
    vec_props[np.isnan(vec_props)] = 0
    vec_props[np.isinf(vec_props)] = 0
    
    sample_data = xr.Dataset({'value_properties': (['label', 'property'], val_props),
                              'vector_properties': (['label', 'property_vec', 'dimension'], vec_props)},
                            coords = {'label': labels,
                                      'property': properties_val,
                                      'property_vec': properties_vect,
                                      'dimension': np.arange(max_vec_dim)})
    sample_data.attrs = dyn_data.attrs
    
    
    sample_data.to_netcdf(os.path.join(data_path, new_filename))
    
#    c=c+1
    
     
    
    
    
    
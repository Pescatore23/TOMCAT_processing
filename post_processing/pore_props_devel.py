# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:36:15 2019

@author: firo
"""

import numpy as np
from scipy import ndimage
from skimage import measure
#import time

def cylinder_coords(x, y, x0=0, y0=0):
    r = np.sqrt( (x-x0)**2 + (y-y0)**2)
    phi = np.arctan2(y-y0, x-x0)
    return r, phi


def reduced_pore_object(labels, label):
    pore = labels == label
    
    pore = np.uint8(pore)

#    time_0 = time.time()
    
#    print('BB')
    bounding_box = ndimage.find_objects(pore)[0]
#    time_BB = time.time()
#    print(time_BB-time_0) 

    pore_object = pore[bounding_box].copy()
    return pore_object, bounding_box

def get_pore_props(pore_object, bounding_box):

    COM = ndimage.measurements.center_of_mass(pore_object)
    COM = COM + np.array([bounding_box[0].start, bounding_box[1].start, bounding_box[2].start])         
           
    inertia_tensor = measure._moments.inertia_tensor(pore_object)
    in_tens_eig = np.linalg.eigvalsh(inertia_tensor)

    major_axis = 4*np.sqrt(in_tens_eig[-1])
    minor_axis = 4*np.sqrt(in_tens_eig[0])
    
    aspect_ratio = major_axis/minor_axis

    z_extent = pore_object.shape[2]
    areas_2D = np.zeros(z_extent)
    perim_2D = areas_2D.copy()
    COM_2D = np.zeros([z_extent, 2])
    
    for z in range(z_extent):
        cross_section = pore_object[:,:,z]
        areas_2D[z] = np.count_nonzero(cross_section)
        perim_2D[z] = measure._regionprops.perimeter(cross_section)
        COM_2D[z, :] = ndimage.measurements.center_of_mass(cross_section)
        
    shape_factor_2D = areas_2D/perim_2D**2
    
    median_shape_factor = np.median(shape_factor_2D)
    mean_shape_factor = np.mean(shape_factor_2D)
    shape_factor_std = shape_factor_2D.std()

    vert_axis = COM_2D.mean(axis=0)
    
    x = COM_2D[:,0]
    y = COM_2D[:,1]
    z = np.arange(z_extent)
    
    try:
        x_fit = np.polyfit(z, x, 1)
        y_fit = np.polyfit(z, y, 1)
        fit_state = 'good'
    except:
        x_fit = [0, 0]
        y_fit = [0, 0]
        fit_state = 'bad'
    
    x_fun = np.poly1d(x_fit)
    y_fun = np.poly1d(y_fit)
    
    tilt_axis = [x_fit[0], y_fit[0]]  #slopes of x(z) and y(z) # ist das nicht die major axis im inertia_tensor? ..., nein!, weil die Pore nicht immer gleich doick ist
    
    rho_v, phi_v = cylinder_coords(COM_2D[:,0], COM_2D[:,1], vert_axis[0], vert_axis[1])
    rho_t, phi_t = cylinder_coords(COM_2D[:,0], COM_2D[:,1], x_fun(z), y_fun(z))
    
    v_ecc = rho_v.std()
    t_ecc = rho_t.std()
    
    dx = np.diff(x)
    dy = np.diff(y)
    
    dir_length = np.sqrt( (x[0]-x[-1])**2 + (y[0]-y[-1]**2)+ z_extent**2)
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
               'tilt_axis_state': fit_state,
               'tilt_axis': tilt_axis,
               'tilt_angle_grad': np.arctan(np.mean([np.abs(tilt_axis)]))*180,
               'eccentricity_from_vertical_axis': v_ecc,
               'eccentricity_from_tilt_axis': t_ecc,
               'distance_end_to_end': dir_length,
               'arc_length': arc_length,
               'tortuosity': tort
            
            }
    
    return results
    
    
    

    
     
    
    
    
    
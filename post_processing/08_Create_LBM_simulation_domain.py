# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:17:39 2019

@author: firo
"""

import xarray as xr
import numpy as np
from scipy import ndimage
import skimage.morphology as skmorph
from skimage.transform import resize
from skimage.morphology import ball
from skimage.measure import label as sklabel

#from joblib import Parallel, delayed
#import multiprocessing as mp


# FIXME include creation of PNM by cutting the void space at defined heights


"""INPUT"""

domain_height = 0.1E-3 #m
LBM_lattice_spacing = 2.75E-6 #m
R = 55E-6/2 #m

pore_label = 27
single_pore = False

PNM = True
PNM_DZ = 50E-6 #m

top_position = 0 #m, measured from top

destination = ''

dest_PNM = ''

fit_file = r"W:\Robert_TOMCAT_3\T3_025_3_III\06_fiber_tracing\T3_025_3_III_fiber_fit_data.nc"
exp_file = r"C:\Zwischenlager\Dyn_Data_1200\dyn_data_T3_025_3_III.nc"


"""END INPUT"""

fitdata = xr.load_dataset(fit_file)
expdata = xr.load_dataset(exp_file)


vx = 2.75E-6 #m
H = domain_height
lx = LBM_lattice_spacing
z0 = int(top_position/vx)
z1 = min(int(H/vx), 2016)
scaling = vx/lx


if single_pore == True:
            
    #find bounding box of ROI
    pore_raw = expdata['label_matrix'][:,:,z0:z1] == pore_label
    pore_raw = ndimage.morphology.binary_dilation(pore_raw, structure = ball(2))
    #
    bounding_box = ndimage.find_objects(pore_raw)[0]
    pore_object = pore_raw[bounding_box].data
    
    
    x_min = bounding_box[0].start
    x_max = bounding_box[0].stop
    y_min = bounding_box[1].start
    y_max = bounding_box[1].stop
    
    DX = int(scaling*(x_max-x_min))
    DY = int(scaling*(y_max-y_min))
    DZ = int(scaling*(z1-z0))
    
    
    domain = np.ones([DX, DY, DZ], dtype = np.bool)
    pore = resize(pore_object, domain.shape).astype(np.bool)
    
    
    # FIXME: Think of how two minimum distance in 3D, cuts through cylinders are not always circles
    
    numfibers = len(fitdata['fiber'])
    
    for z in range(domain.shape[2]):
        
        z_lx = z + int(z0*vx/lx)
        
        for fiber in range(numfibers):
            x_fun = np.poly1d(fitdata['trace_fits'][1, :, fiber])
            y_fun = np.poly1d(fitdata['trace_fits'][0, :, fiber])
            
            
            x0_lx = vx/lx * (x_fun(z_lx*lx/vx)-x_min)
            y0_lx = vx/lx * (y_fun(z_lx*lx/vx)-y_min)
            
            if x0_lx < -R/lx: continue
            if y0_lx < -R/lx: continue
        
            if x0_lx > DX+R/lx: continue
            if y0_lx > DY+R/lx: continue
        
            for x in range(domain.shape[0]):
                for y in range(domain.shape[1]):
                    if (x-x0_lx)**2+(y-y0_lx)**2 < (R/lx)**2:
                        domain[x,y,z] = False
                        
    sim_domain = np.bitwise_and(pore, domain).astype(int)
    
#    sim_domain.tofile(destination)
        

if PNM == True:
    dz_pnm = int(PNM_DZ/lx)
    
    void = expdata['label_matrix'][:,:,z0:z1]
#    void = ndimage.morphology.binary_dilation(void, structure = ball(2)).astype(np.bool)
    
    DX = int(scaling*void.shape[0])
    DY = int(scaling*void.shape[1])
    DZ = int(scaling*void.shape[2])   
    
    domain = np.zeros([DX, DY, DZ], dtype = np.bool)
    hull = np.zeros([DX, DY, DZ], dtype = np.bool)
#    void = resize(void, domain.shape).astype(np.bool)

    numfibers = len(fitdata['fiber'])
    
    for z in range(domain.shape[2]):
        
        z_lx = z + int(z0*vx/lx)
        
        for fiber in range(numfibers):
            x_fun = np.poly1d(fitdata['trace_fits'][1, :, fiber])
            y_fun = np.poly1d(fitdata['trace_fits'][0, :, fiber])
            
            
            x0_lx = vx/lx * x_fun(z_lx*lx/vx)
            y0_lx = vx/lx * y_fun(z_lx*lx/vx)
        
            for x in range(domain.shape[0]):
                for y in range(domain.shape[1]):
                    if (x-x0_lx)**2+(y-y0_lx)**2 < (R/lx)**2:
                        domain[x,y,z] = True
        hull[:,:,z] = skmorph.convex_hull_image(domain[:,:,z])
        
    sim_domain = np.bitwise_xor(hull, domain)
           
    
    for z in range(sim_domain.shape[2]):
        sim_domain[:,:,z] = skmorph.remove_small_objects(sim_domain[:,:,z], min_size=5, connectivity=1)
    void = None
    domain = None
#    from here do pore segmentation, i.e. connected components after cutting in dz_pnm thick slices

    pnm_domain = np.zeros(sim_domain.shape, dtype = np.uint16)
    
    num_segments = int(sim_domain.shape[2]/dz_pnm) + 1
    ref_color = 0
    
    for i in range(num_segments):
        z_pnm_0 = i*dz_pnm
        if z_pnm_0 > pnm_domain.shape[2]-1: continue
    
        z_pnm_1 = min(z_pnm_0 + dz_pnm, pnm_domain.shape[2])
        
        test_volume = sim_domain[:,:,z_pnm_0:z_pnm_1]
        label = sklabel(test_volume, connectivity = 2)
        num_pores = label.max()
        label[np.where(label>0)] = label[np.where(label>0)] + ref_color
        pnm_domain[:, :, z_pnm_0:z_pnm_1] = label
        ref_color = ref_color + num_pores
        
#    pnm_domain.tofile(dest_PNM)
        
        
        

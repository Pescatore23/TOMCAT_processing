# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:50:19 2019

@author: firo
"""

import sys
library=r"R:\Scratch\305\_Robert\Python_Library"

if library not in sys.path:
    sys.path.append(library)
    
    
    
import RobPyLib
import os
import xarray as xr
import numpy as np
import skimage.morphology as skmorph
from skimage import io

from joblib import Parallel, delayed
import multiprocessing as mp

num_cores = mp.cpu_count()

drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_3')
expDataPath = r"C:\Zwischenlager\Dyn_Data_1200"

R=int(54.5E-6/2.75E-6/2)


traceFolder = '06_fiber_tracing'


samples = os.listdir(baseFolder)


def create_model(sample, baseFolder = baseFolder, expDataPath = expDataPath):
    
    flag = True
    targetFolder = os.path.join(baseFolder, sample, '04_c_model_void')
        
    fit_file = os.path.join(baseFolder, sample, traceFolder, ''.join([sample,'_fiber_fit_data.nc']))
#    exp_file = os.path.join(expDataPath, ''.join(['dyn_data_',sample,".nc"]))
    
    if os.path.exists(fit_file):
        fitdata = xr.load_dataset(fit_file)
#        expdata = xr.load_dataset(exp_file)
        
#        shp = expdata['transition_matrix'].shape
        names = os.listdir(os.path.join(baseFolder, sample, '02a_temporal_mean'))
        if 'Thumbs.db' in names: names.remove('Thumbs.db')
        
        im = io.imread(os.path.join(baseFolder, sample, '02a_temporal_mean', names[0]))
        
        shp = [im.shape[0], im.shape[1], len(names)]
        
#        if os.path.exists(targetFolder):
#            if len(os.listdir(targetFolder)) > shp[2]-10:
#                flag = False
        
        if flag:
            domain = np.zeros(shp, dtype = np.bool)
            hull = np.zeros(shp, dtype = np.bool)
            
            numfibers = len(fitdata['fiber'])
            
            for z in range(domain.shape[2]):
                
                for fiber in range(numfibers):
                    x_fun = np.poly1d(fitdata['trace_fits'][1, :, fiber])
                    y_fun = np.poly1d(fitdata['trace_fits'][0, :, fiber])
                    
                    
                    x0 = x_fun(z)
                    y0 = y_fun(z)
                
                    for x in range(domain.shape[0]):
                        for y in range(domain.shape[1]):
                            if (x-x0)**2+(y-y0)**2 < R**2:
                                domain[x,y,z] = True
                hull[:,:,z] = skmorph.convex_hull_image(domain[:,:,z])
                
            sim_domain = np.bitwise_xor(hull, domain)
                   
            
            for z in range(sim_domain.shape[2]):
                sim_domain[:,:,z] = skmorph.remove_small_objects(sim_domain[:,:,z], min_size=5, connectivity=1)
            
            sim_domain = np.uint8(sim_domain)*255
            
            if not os.path.exists(targetFolder):
                os.mkdir(targetFolder)
            RobPyLib.CommonFunctions.ImportExport.WriteStackNew(targetFolder, names, sim_domain)
    

Parallel(n_jobs=num_cores)(delayed(create_model)(sample) for sample in samples) 
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 09:14:36 2020

@author: firo
"""

import robpylib
import os
import xarray as xr
# import dask.array
# from dask.distributed import Client
import joblib
from joblib import Parallel, delayed
import numpy as np

workers = 16
# client = Client(processes=False)             # create local cluster
# client = Client("scheduler-address:8786")  # or connect to remote cluster


drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_3')
resultFolder = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives', 'reprocessing_previous_frame_as_register_reference')

rawbaseFolder = r"W:\disk2"

repeats = robpylib.TOMCAT.INFO.samples_to_repeat

def process_slice(z, sourceFolder, matFolder, fiber_folder, repeat=False):
    Tstack, names, scans = robpylib.CommonFunctions.ImportExport.OpenTimeStack(sourceFolder, z)
    if repeat:
        Tstack = Tstack[:,:,4:]
    Tstack = robpylib.TOMCAT.processing.intensity_correction(Tstack)
    Tstack, transmat = robpylib.TOMCAT.processing.register(Tstack, z, matFolder, return_mat = True)
    transitions, transitions2 = robpylib.TOMCAT.processing.segmentation(Tstack, fiber_folder, z)
    return transitions, transitions2, transmat



samples = os.listdir(baseFolder)
for sample in samples:
    repeat = False
    if sample in repeats:
        repeat = True
        continue
    print(sample)
    if not sample[1] == '3': continue
    
    sourceFolder = os.path.join(rawbaseFolder, sample, '00_raw')
    
    if not os.path.exists(sourceFolder): continue
    print(sample)
    fiber_folder = os.path.join(baseFolder, sample, '01a_weka_segmented_dry', 'classified')
    
    matFolder = os.path.join(resultFolder, 'registration_matrices')
    if not os.path.exists(matFolder):
        os.mkdir(matFolder)
        
    # with joblib.parallel_backend('dask'):
    result=Parallel(n_jobs=workers)(delayed(process_slice)(z, sourceFolder, matFolder, fiber_folder, repeat = repeat) for z in range(2016))
    
    result = np.array(result)
    
    transitions = np.zeros((result[0,0].shape[0],result[0,0].shape[1], result.shape[0]), dtype = np.uint16)
    transitions2 = transitions.copy()                     
    transmats = np.zeros((result[0,2].shape[0], result[0,2].shape[1], result[0,2].shape[2],result.shape[0]), dtype=result[0,2].dtype)
    
    for z in range(result.shape[0]):
        transitions[:,:,z] = result[z,0]
        transitions2[:,:,z] = result[z,1]
        transmats[:,:,:,z] = result[z,2]
    
    time = robpylib.TOMCAT.TIME.TIME[sample]
    
    if repeat: time = time[4:]
    tension = np.uint16(sample[3:6])
    sample_id = np.uint8(sample[7])
    data = xr.Dataset({'transition_matrix': (['x','y','z'], transitions),
                       'transition_2_matrix': (['x','y','z'], transitions2),
                       'tranmat': (['time','p1','p2','z'], transmats)},
                      coords = {'x': np.arange(transitions.shape[0]),
                                'y': np.arange(transitions.shape[1]),
                                'z': np.arange(transitions.shape[2]),
                                'p1': np.arange(transmats.shape[0]),
                                'p2': np.arange(transmats.shape[0]),
                                'time': time},
                      attrs = {'name': sample,
                               'sample_id': sample_id,
                               'tension': tension,
                               'repeat': repeat,
                               'voxel_size': '2.75 um',
                               'voxel': 2.75E-6} #m
        )
    data['transition_matrix'].attrs['units'] = 'time step'
    data['transition_2_matrix'].attrs['units'] = 'time step'   
    
    filename = os.path.join(resultFolder, ''.join([sample, '_segmented.nc']))
    data.to_netcdf(filename)
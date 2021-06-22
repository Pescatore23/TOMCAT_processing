# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:05:56 2021

@author: firo
"""

import skimage.measure
import skimage.segmentation
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import os
import robpylib
import xarray as xr

temp_folder= r"Z:\users\firo\joblib_tmp"
drive = r"A:"
baseFolder = os.path.join(drive, 'Robert_TOMCAT_4')
netcdfFolder = os.path.join(drive, 'Robert_TOMCAT_4_netcdf4_split_v2')

def get_perimeter(fiber_slice):
    #use crofton perimeter, it is much better -> learn what that is
    im = fiber_slice
    perim = skimage.measure.perimeter_crofton(im)
    return perim

# do this also for each fiber separately and compare because fibers in contact are counted as one solid object and perimeter will be under-estimated
def get_perimeter_profile(fibers, num_cores=16, temp_folder = temp_folder):
    zmax = fibers.shape[2]
    result = Parallel(n_jobs=num_cores, temp_folder = temp_folder)(delayed(get_perimeter)(fibers[:,:,z]) for z in range(zmax))
    perim_prof = np.array(result)
    return perim_prof

def make_skeleton(matrix, point_list, points, value):
    for point in point_list:
        y = points['X Coord'][point]
        x = points['Y Coord'][point]
        z = points['Z Coord'][point]
        matrix[x,y,z] = value
    return matrix

def get_perimeter_profile_separated(fibers, path, num_cores=16, fiber_buffer = 70):
    segments = pd.read_excel(path, sheet_name = 2)
    points = pd.read_excel(path, sheet_name = 1)
    # nodes = pd.read_excel(path, sheet_name = 0)
    markers = np.zeros(fibers.shape, dtype = np.uint8)
    j = 1
    for segment in segments['Segment ID']:
        segment_points = segments['Point IDs'][segment]
        segment_points = np.array([int(i) for i in segment_points.split(',')])
        markers = make_skeleton(markers, segment_points, points, j)    
        
        j = j + 1
    
    print('marker based watershed inside mask')
    labeled = skimage.segmentation.watershed(fibers, markers=markers, mask=fibers)
    
    perim_profiles = np.zeros((fiber_buffer,fibers.shape[2]))
    print('get perimeter profile for each fiber')
    for k in range(j-1):
        perim_profiles[k,:] = get_perimeter_profile(labeled == k+1)
    
    
    return labeled, perim_profiles

samples = os.listdir(baseFolder)
if '.DS_Store' in samples: samples.remove('.DS_Store')

c = 0
sl = len(samples)
fiber_buffer = 70
fiber_perim_profiles = np.zeros((sl, fiber_buffer, 2016))
full_perim_profiles = np.zeros((sl, 2016))
for sample in samples:
    print(str(c+1),'/',str(sl),' :',sample)
    
    sampleFolder = os.path.join(baseFolder, sample)
    fiberFolder = os.path.join(sampleFolder, '01a_weka_segmented_dry', 'classified')
    labeledfiberFolder = os.path.join(sampleFolder, '06_fiber_tracing', 'labeled_fibers')
    tracingFolder = os.path.join(sampleFolder, '06_fiber_tracing')
    fiberpath = os.path.join(tracingFolder, ''.join([sample,'.CorrelationLines.xlsx']))
    
    if not os.path.exists(fiberpath):
        print(sample,' has no fiber traces, skipping ...')
        c = c + 1
        continue
    
    if not os.path.exists(labeledfiberFolder):
        os.mkdir(labeledfiberFolder)
    
    print('loading fibers')
    fibers, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiberFolder, track=False)
        
    print('calculate image perimeter profile')
    full_perim_profiles[c,:] = get_perimeter_profile(fibers)
    
    print('calculated the perimeter of all fibers and add up')
    labeled, perim_profiles = get_perimeter_profile_separated(fibers, fiberpath, fiber_buffer=fiber_buffer)
    fiber_perim_profiles[c,:,:] = perim_profiles
    
    print('save labeled fibers to disk')
    robpylib.CommonFunctions.ImportExport.WriteStackNew(labeledfiberFolder, names, labeled)
    
    c = c + 1
    
    
data = xr.Dataset({'full_perimeter': (['sample', 'z'], full_perim_profiles),
                   'fiber_perimeters': (['sample', 'fiber', 'z'], fiber_perim_profiles)},
                  coords = {'sample': samples,
                            'z' : np.arange(2016),
                            'fiber': np.arange(fiber_buffer)+1})

data['z'].attrs['unit'] = 'px'
data['full_perimeter'].attrs['unit'] = 'px'
data['fiber_perimeters'].attrs['unit'] = 'px'

data.to_netcdf(os.path.join(netcdfFolder, 'perimeter_profiles.nc'))
    
    
    



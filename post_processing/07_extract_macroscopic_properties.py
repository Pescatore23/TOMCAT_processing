# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:33:47 2019

@author: firo
"""

import sys
library=r"R:\Scratch\305\_Robert\Python_Library"

if library not in sys.path:
    sys.path.append(library)
    
    
import RobPyLib
import numpy as np
import os
import xarray as xr

excluded_samples=[
 'T3_100_4',        #FOV moved during acquisition
 'T3_300_6',        # water only in little pores
 'T3_300_3_III',    #no water
 'T4_025_3_III',    #wet from start
 'T3_300_8',        #FOV moved
 'T3_025_7',        #water only in small pores
 'T3_025_4_III',    #little water, data set incomplete, missing time steps reconstructed at PSI-ra-cluster, but only little water -> not reasonable to process?!
 'T3_025_3_II'      #little, water wet with acetone
 ]

baseFolder1 = r"Z:\Robert_TOMCAT_3"
baseFolder2 = r"Z:\Robert_TOMCAT_3_Part_2"

samples1 = os.listdir(baseFolder1)
samples2 = os.listdir(baseFolder2)

samples = samples1 + samples2

data_path = r"H:\11_Essential_Data\03_TOMCAT\08_Macroscopic_Properties"

porosity = np.zeros([len(samples1)+len(samples2), 2016])
porespace = np.zeros([len(samples1)+len(samples2), 2016])
solid = np.zeros([len(samples1)+len(samples2), 2016])

c = 0


for sample in samples1:
    baseFolder = baseFolder1
    fibers, _ = RobPyLib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '01a_weka_segmented_dry', 'classified'))
    fibers = fibers/fibers.max()
    
    voids, _ = RobPyLib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '04a_void_space'))
    voids = voids/voids.max()
    
    
    fibervolume = fibers.sum(axis = 0).sum(axis = 0)
    voidvolume = voids.sum(axis = 0).sum(axis = 0)
    volume = fibervolume + voidvolume
    porosity[c, :] = voidvolume/volume
    solid[c, :] = fibervolume
    porespace[c, :] = voidvolume
    
    c= c +1 
    
for sample in samples2:
    if sample in excluded_samples:
        c=c+1
        continue
    baseFolder = baseFolder2
    fibers, _ = RobPyLib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '01a_weka_segmented_dry', 'classified'))
    fibers = fibers/fibers.max()
    
    voids, _ = RobPyLib.CommonFunctions.ImportExport.ReadStackNew(os.path.join(baseFolder, sample, '04a_void_space'))
    voids = voids/voids.max()
    
    
    fibervolume = fibers.sum(axis = 0).sum(axis = 0)
    voidvolume = voids.sum(axis = 0).sum(axis = 0)
    volume = fibervolume + voidvolume
    porosity[c, :] = voidvolume/volume
    solid[c, :] = fibervolume
    porespace[c, :] = voidvolume
    
    c= c +1 
    
    
data = xr.Dataset({'porosity': (['sample', 'z'], porosity),
                   'fiber_volume': (['sample', 'z'], solid),
                   'void_volume': (['sample', 'z'], porespace)},
                    coords = {'sample': samples,
                              'z': np.arange(2016)*2.75E-6},
                              attrs = {'name': 'porosity data of all samples'})

data['z'].attrs['units'] = 'm'
data['fiber_volume'].attrs['units'] = 'vx'
data['void_volume'].attrs['units'] = 'vx'


data.to_netcdf(os.path.join(data_path, 'TOMCAT3_porosits.nc'))
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:56:56 2019

Preparations necessary to run this script:
    
    
    connect via VPN to Empa network
    
    connect to "R" network drive
    
    set up a dask-scheduler (usually on one powerful machine) (via ssh(only linux and some win10) or remote desktop)
        type in conda command line:
            dask-scheduler
            remember INFO of scheduler at: ...
        
        in a new terminal type:
            dask-worker tcp://<scheduler-ip>:<port>   port is usually 8786
        repeat this for every machine you want to add to the new "cluster"
       to run this script, every machine needs also to be logged in on the NAS
        
        
    define a suitable chunksize:
        upper limit roughly: RAM/threads/2.5
        guideline:
            as big as possible, as small as necessary
            too small --> too much overhead
            too big --> inefficient memory usage, workes may sit idle while waiting for parallel results
            
    calculation can be surveiled by typing <scheduler-ip>:8787/status in any browser of any machine in empa vpn


@author: firo
"""
import sys
library=r"R:\Scratch\305\_Robert\Python_Library"

if library not in sys.path:
    sys.path.append(library)
#import xarray as xr
#import dask
#import numpy as np
import netCDF4

import dask.array as da
from dask.distributed import Client
import dask.config
import RobPyLib
import os
import numpy as np

clientlocation = '152.88.86.193:8786'    #<scheduler-ip>:<port>

chunksize = 35 * 128 # needs to be multiple of 128Mb (default chunk size), but this is just an upper limit. Probably not that strict to use multiples
chunkstring = ''.join([str(chunksize), 'MiB'])



dask.config.set({'array.chunk_size': chunkstring}) 
client = Client(clientlocation, processes = False)
#client = Client(processes = False)

network_location = '//152.88.86.87/data118/'    #probably only works on Windows
destinationBase = os.path.join(network_location, "Robert_TOMCAT_3_Part_2")

samples = os.listdir(destinationBase)



c=0
for sample in samples:
    c=c+1
    print(sample, '(',c,'/',len(samples),')')
#    if c<3: continue
#    path = r"Z:\Robert_TOMCAT_3_netcfd4_archives\02_registered\registeredT3_025_3_III.nc"
    path = os.path.join(network_location, r"Robert_TOMCAT_3_netcfd4_archives_1_34TB\02_registered", ''.join(['registered',sample,'.nc']))
    targetFolder = os.path.join(destinationBase, sample, '02a_temporal_mean')
    
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)
    
    if len(os.listdir(targetFolder))>2000:
        print('already calculated, skipping...')
        continue
    
    if not os.path.exists(path):
        print('no netCDF4 version yet available, skipping...')
        continue
    data = netCDF4.Dataset(path)['registered']

    data_dask = da.from_array(data, chunks='auto')

    temporal_mean_fun = data_dask.mean(axis=0)
    
    temporal_mean = temporal_mean_fun.compute()
    temporal_mean = np.float32(temporal_mean)

    refpath = os.path.join(destinationBase, sample, '02_pystack_registered')
    refFolder = os.listdir(refpath)[0]
    names = os.listdir(os.path.join(refpath, refFolder))
    if 'Thumbs.db' in names: names.remove('Thumbs.db')
    
    RobPyLib.CommonFunctions.ImportExport.WriteStackNew(targetFolder, names, temporal_mean)

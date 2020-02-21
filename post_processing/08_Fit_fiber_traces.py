# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:25:28 2019

@author: firo
"""

#import sys
#library=r"R:\Scratch\305\_Robert\Python_Library"
#
#if library not in sys.path:
#    sys.path.append(library)
#    
#    
#import RobPyLib
import os
import numpy as np
import pandas as pd
import xarray as xr

# FIXME: automate, attention: not all samples have 32 fibers, some more or less!!


drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_3')
traceFolder = '06_fiber_tracing'
dataFolder = r"H:\11_Essential_Data\03_TOMCAT\07_TOMCA3_dynamic_Data"

samples = os.listdir(baseFolder)


#flag = False

for sample in samples:
#    if flag: continue
    if sample[1] == '4': continue
    trace_file = os.path.join(baseFolder, sample, traceFolder, ''.join([sample,'.CorrelationLines.xlsx']))
#    see if pandas can read this like xlsx or if conversion is needed. Yes, most simple is to convert it to xlsx by loading it with excel and save
    
    points = pd.read_excel(trace_file, sheet_name = 'Points')

    seps = np.where(np.diff(points['Z Coord'])<0)[0]
    num_fibers = len(seps)+1

    deg = 10
    coeff = np.zeros([2, deg+1, num_fibers], dtype = np.float64)
    vx = 2.75E-6 #m


    #fit first fiber

    x = points['X Coord'][:seps[0]+1]
    y = points['Y Coord'][:seps[0]+1]
    z = points['Z Coord'][:seps[0]+1]

    x_fit = np.polyfit(z, x, deg)
    y_fit = np.polyfit(z, y, deg)

    coeff[0, :, 0] = x_fit
    coeff[1, :, 0] = y_fit


    #fit fibers

    for i in range(1,num_fibers-1):
        x = points['X Coord'][seps[i-1]+1:seps[i]+1]
        y = points['Y Coord'][seps[i-1]+1:seps[i]+1]
        z = points['Z Coord'][seps[i-1]+1:seps[i]+1]
        
        x_fit = np.polyfit(z, x, deg)
        y_fit = np.polyfit(z, y, deg)
        
        coeff[0, :, i] = x_fit
        coeff[1, :, i] = y_fit
    
    
    #fit last fiber
    
    x = points['X Coord'][seps[-1]+1:]
    y = points['Y Coord'][seps[-1]+1:]
    z = points['Z Coord'][seps[-1]+1:]
    
    x_fit = np.polyfit(z, x, deg)
    y_fit = np.polyfit(z, y, deg)
    
    coeff[0, :, -1] = x_fit
    coeff[1, :, -1] = y_fit



    dataset = xr.Dataset({'trace_fits': (['coord', 'coeff', 'fiber'], coeff)},
                            coords = {'coord': ['y','x'],                       #attention: x & y swapped to be consistend with pore matrix
                                      'coeff': coeff.shape[1]-np.arange(coeff.shape[1]),
                                      'fiber': np.arange(1, coeff.shape[2]+1),},
                                      attrs = {'name' : sample,
                                               'comment': 'fit of fiber center coordinates: x,y(z[m]) = vx*(kn*(z/vx)^n + ... k1*(z/vx) + k0)',
                                               'pixel size (vx)': '2.75 um',
                                               'waterline': 'reasonable z-range: 0-1300vx'})
    
#    data_path = r"H:\03_Besprechungen\Simulation_Correspondence\Jianlin Zhao"
    data_file = os.path.join(baseFolder, sample, traceFolder, ''.join([sample,'_fiber_fit_data.nc']))
    
    dataset.to_netcdf(data_file)                    

#    flag = True


#test_image = np.zeros([288,288,2016], dtype = np.uint8)
#
#
#D = 55E-6  #m
#vx = 2.5E-6 #m
#
#r = D/vx/2-0.5
#r2 = r**2
#
#y_fit = np.poly1d(coeff[0, :, 0])
#x_fit = np.poly1d(coeff[1, :, 0])
#
#for z in range(2016):
#    x0 = x_fit(2016-z)
#    y0 = y_fit(2016-z)
#    for x in range(288):
#        for y in range(288):
#            if (x-x0)**2+(y-y0)**2 < r2:
#                test_image[x,y,-z-1] = 255
#                
#test_path = r"T:\Samples_with_Water_inside\32_200_025H2_cont\04_fiber_tracing\fiber_fit_test_image"
#names = os.listdir(r"T:\Samples_with_Water_inside\32_200_025H2_cont\01a_weka_segmented_dry\classified")
#
#
#RobPyLib.CommonFunctions.ImportExport.WriteStackNew(test_path, names, test_image)


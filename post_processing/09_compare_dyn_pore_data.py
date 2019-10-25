# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:21:58 2019

@author: firo
"""
#import sys
#library=r"R:\Scratch\305\_Robert\Python_Library"
#
#if library not in sys.path:
#    sys.path.append(library)


import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import robpylib
from joblib import Parallel, delayed
import multiprocessing as mp

num_cores = mp.cpu_count()

drive = '//152.88.86.87/data118'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1400_dry_seg_aniso_sep'

sourceFolder  = os.path.join(data_path, processing_version)
plot_folder = r"R:\Scratch\305\_Robert\plots_pore_prop_corr"

all_together = os.path.join(plot_folder, 'all')
if not os.path.exists(all_together):
    os.mkdir(all_together)
    
tension_sorted = os.path.join(plot_folder, 'tension')
if not os.path.exists(tension_sorted):
    os.mkdir(tension_sorted)
    
sample_sorted = os.path.join(plot_folder, 'sample')
if not os.path.exists(sample_sorted):
    os.mkdir(sample_sorted)

samples_crude = os.listdir(sourceFolder)

samples = []
#excluded_samples = ['T3_025_1', 'T3_100_1', 'T3_100_10', 'T3_300_15', 'T3_300_9_III']   #populate with samples where segmentation failed


good_samples = robpylib.TOMCAT.INFO.good_samples
# samples with no processing errors

parameter = ['filling start time step', 'filling end time step','filling start [s]', 'filling end [s]', 'max filling rate [vx_s]', 'time step of max filling rate', 'median filling rate [vx_s]', 'num filling peaks', 'final water volume [vx]', 'final_saturation']

for sample in samples_crude:
    if sample[:3] == 'dyn':
        samples.append(sample)
        
data_dict = {}

c=0
c25 = 0
c100 = 0
c300 = 0
for sample in samples:
#    if c>1: continue
    if not sample[9:-3] in good_samples: continue
    c=c+1
    time_list = robpylib.TOMCAT.TIME.TIME[sample[9:-3]]
    dyn_data = xr.load_dataset(os.path.join(sourceFolder, sample))
    pore_data = xr.load_dataset(os.path.join(sourceFolder, ''.join(['pore_props_', sample[9:]])))
    data = dyn_data.merge(pore_data)
    data['parameter'] = parameter
#    data.coords['sig_fit_var'] = ['t0 [s]', 'beta [1_s]', 'alpha [vx]', 'R2']
    
    

    test_data = np.concatenate([data['dynamics'], data['fit_data'], data['sig_fit_data'], data['value_properties'], ], axis = 1)
    test_data = np.concatenate([test_data, np.array([test_data[:,9]/test_data[:,8],]).transpose()], axis = 1) # rel_slope = slope/volume
    
    steps = test_data[:,5].copy().astype(np.uint16)
    time_max = np.array([time_list[step] for step in steps])
    
    test_data = np.concatenate([test_data, np.array([time_max,]).transpose()], axis = 1)
    
    norm_a_r = test_data[:,15]/np.sqrt(test_data[:,29])
    test_data = np.concatenate([test_data, np.array([norm_a_r,]).transpose()], axis = 1)
    
    variables = list(data['parameter'].data) + list(data['fit_var'].data) + list(data['sig_fit_var'].data) + list(data['property'].data) + ['rel_slope']+ ['max_filling_time [s]'] + ['norm_aspect_ratio']

    data_dict[sample[9:-3]] = test_data.copy()   

#    all samples together    
    if c==1:
        combined_data = test_data
    if c>1:
        combined_data = np.concatenate([combined_data, test_data], axis = 0)
##########################


    # combined datasets per tension value
    if data.attrs['tension'] == 25:
        c25 = c25 + 1  
        if c25 == 1:
            combined_025 = test_data   
        if c25 > 1:
            combined_025 = np.concatenate([combined_025, test_data], axis = 0)
    
    if data.attrs['tension'] == 100:
        c100 = c100 + 1  
        if c100 == 1:
            combined_100 = test_data   
        if c100 > 1:
            combined_100 = np.concatenate([combined_100, test_data], axis = 0)
            
    if data.attrs['tension'] == 300:
        c300 = c300 + 1  
        if c300 == 1:
            combined_300 = test_data   
        if c300 > 1:
            combined_300 = np.concatenate([combined_300, test_data], axis = 0)
#  samples         

combined_data[np.isnan(combined_data)] = 0
combined_data[np.isinf(combined_data)] = 0

combined_025[np.isnan(combined_025)] = 0
combined_025[np.isinf(combined_025)] = 0

combined_100[np.isnan(combined_100)] = 0
combined_100[np.isinf(combined_100)] = 0

combined_300[np.isnan(combined_300)] = 0
combined_300[np.isinf(combined_300)] = 0

# #values that cover whole dataset
ylims = {'filling start time step': (0, 360),
         'filling end time step': (0, 360),
         'filling start [s]': (0, 1750),
         'filling end [s]': (0, 1750),
         'max filling rate [vx_s]': (0, 50000), #(0, 5E9),
         'time step of max filling rate': (0, 360),
         'median filling rate [vx_s]': (0, 20), #(0, 120),
         'num filling peaks': (0, 20), #(0, 35),
         'pore volume [vx]': (0, 1000000), # (0, 2500000),
         'slope': (0, 4000), #(0, 30000),
         'rel_slope': (0,0.2), #(0,1),
         'onset': (0, 1800), #(0, 5E18),
         't0 [s]': (0, 1750), # (-80000, 50000),
         'beta [1_s]': (0,12), # (0, 600),
         'alpha [vx]': (0, 1000000), #(0, 7E9),
         'R2': (0, 1.05),
         'major_axis': (0, 1800),
         'minor_axis': (0, 170),
         'aspect_ratio': (0,50), #(0, 90),
         'median_shape_factor': (0, 0.2),
         'mean_shape_factor': (0, 0.2), #(0, 0.8),
         'shape_factor_std': (0, 0.8), #(0, 1.6),
         'shp_fac_rel_var': (0,1), #(0,3),
         'tilt_axis_state': (-0.05, 1.05),
         'tilt_angle_grad': (0, 50), #(0, 80),
         'eccentricity_from_vertical_axis': (0, 25),
         'eccentricity_from_tilt_axis': (0, 16),
         'distance_end_to_end': (0, 1500),
         'arc_length': (0, 1750),
         'tortuosity': (0.95, 1.5), #(0.95, 3),
         'volume': (0, 1000000),
         'mean_area': (0, 500),# (0, 2750),
         'median_area': (0, 500), #(0,3000),
         'area_std': (0,500), #(0,1750),
         'area_rel_var': (0, 1.75),
         'max_filling_time [s]': (0, 1750),
         'norm_aspect_ratio': (0,9), #(0,15),
         'angular_dist': (0,100),#(0,360)
         'final_saturation': (0,1),
         'final water volume [vx]': (0, 200000)
         }

def plotting(i,j, combined_data=combined_data, combined_025=combined_025,
             combined_100=combined_100, combined_300=combined_300, data_dict=data_dict):
#    FIXME: it is necessary to plot each sample in different colors
#    plot different tension values in different colors
    
#    plot all together
        plt.figure()
        plt.plot(combined_data[:,j], combined_data[:,i], 's', markersize = 1)
        plt.title(variables[i])
        plt.xlabel(variables[j])
        plt.xlim(ylims[variables[j]])
        plt.ylim(ylims[variables[i]])
        filename = os.path.join(all_together,''.join([variables[j],'___',variables[i],'.png']))
        plt.savefig(filename, dpi=500, bbox_inches = 'tight')
        plt.close()   
        
#    plot tensions in different colors
        plt.figure()
        plt.plot(combined_025[:,j], combined_025[:,i], 's', markersize = 1)
        plt.plot(combined_100[:,j], combined_100[:,i], 's', markersize = 1)
        plt.plot(combined_300[:,j], combined_300[:,i], 's', markersize = 1)
        plt.title(variables[i])
        plt.xlabel(variables[j])
        plt.xlim(ylims[variables[j]])
        plt.ylim(ylims[variables[i]])
        filename = os.path.join(tension_sorted,''.join([variables[j],'___',variables[i],'.png']))
        plt.savefig(filename, dpi=500, bbox_inches = 'tight')
        plt.close()
        
#    plot samples in different colors
        plt.figure()
        for sample in list(data_dict.keys()):
            plt.plot(data_dict[sample][:,j], data_dict[sample][:,i], 's', markersize = 1)       
        plt.title(variables[i])
        plt.xlabel(variables[j])
        plt.xlim(ylims[variables[j]])
        plt.ylim(ylims[variables[i]])
        filename = os.path.join(sample_sorted,''.join([variables[j],'___',variables[i],'.png']))
        plt.savefig(filename, dpi=500, bbox_inches = 'tight')
        plt.close()    




for j in range(combined_data.shape[1]): 
#    normalize histograms and maybe crop high peasks
#       histogram all
    plt.figure()
    hist = np.histogram(combined_data[:,j][combined_data[:,j]<ylims[variables[j]][1]], bins = 50)
    plt.hist(combined_data[:,j], hist[1], density = True, log = True)
    plt.title(variables[j])
    plt.xlabel(variables[j])
    hist_folder = os.path.join(plot_folder,
                            'histogram')
    if not os.path.exists(hist_folder):
        os.mkdir(hist_folder)
    filename = os.path.join(hist_folder,''.join([variables[j],'.png']))
    plt.savefig(filename, dpi=500, bbox_inches = 'tight')
    plt.close()   
#       histogram tension
    plt.figure()
    hist1 = np.histogram(combined_025[:,j][combined_025[:,j]<ylims[variables[j]][1]], bins =50)
#    plt.hist(combined_025[:,j], hist1[1])
#    hist2 = np.histogram(combined_100[:,j][combined_100[:,j]<ylims[variables[j]][1]], bins =50)
#    plt.hist(combined_100[:,j], hist2[1])
#    hist3 = np.histogram(combined_300[:,j][combined_300[:,j]<ylims[variables[j]][1]], bins =50)
#    plt.hist(combined_300[:,j], hist3[1])
    plt.hist([combined_025[:,j][combined_025[:,j]<ylims[variables[j]][1]],
              combined_100[:,j][combined_100[:,j]<ylims[variables[j]][1]],
              combined_300[:,j][combined_300[:,j]<ylims[variables[j]][1]]], hist1[1], density = True, log = True)
    
    plt.title(variables[j])
    plt.xlabel(variables[j])
    
    hist_ten_folder = os.path.join(plot_folder,
                            'histogram_tension')
    if not os.path.exists(hist_ten_folder):
        os.mkdir(hist_ten_folder)

    filename = os.path.join(hist_ten_folder,''.join([variables[j],'.png']))
    plt.savefig(filename, dpi=500, bbox_inches = 'tight')
    plt.close()      

         
    Parallel(n_jobs=num_cores)(delayed(plotting)(i,j) for i in range(combined_data.shape[1]))
#    for i in range(combined_data.shape[1]):
#        plt.figure()
#        plt.plot(combined_data[:,j], combined_data[:,i], '.')
#        plt.title(variables[i])
#        plt.xlabel(variables[j])
#        plt.xlim(ylims[variables[j]])
#        plt.ylim(ylims[variables[i]])
#        filename = os.path.join(r"R:\Scratch\305\_Robert\plots_pore_prop_corr",
#                                ''.join([variables[j],'___',variables[i],'.png']))
#        plt.savefig(filename, dpi=500, bbox_inches = 'tight')
#        plt.close()
        
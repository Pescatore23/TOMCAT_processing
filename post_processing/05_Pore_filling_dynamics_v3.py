# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 11:33:32 2019

pore scale flow analysis

input: 
    pore label image
    transition image



@author: firo
"""


import scipy as sp
import numpy as np
import os
#import skimage.measure as skm
import matplotlib.pyplot as plt
import robpylib
import xarray as xr
#import time
import scipy.optimize
from joblib import Parallel, delayed
import multiprocessing as mp

parallel = True

num_cores = mp.cpu_count()
#num_cores = 8

time_limit = {'T3_100_10_III': 344,
              'T3_300_5': 229,
              'T3_100_7': 206,
              'T3_100_10': 232}

waterline = 1200
px=2.75E-6 #m
rho=997 #kg/m3

#baseFolder = 'X:\\Samples_with_Water_inside'
#baseFolder = r'U:\TOMCAT_3_segmentation'
#baseFolder = r"Z:\Robert_TOMCAT_3"
vx = 2.75E-6  #m
rho = 997 #kg/m3  density of water
#data_path = r"H:\11_Essential_Data\03_TOMCAT\07_TOMCA3_dynamic_Data"
#data_path = r"C:\Zwischenlager\Dyn_Data"

drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_3')
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives', 'processed_1200_dry_seg_aniso_sep')

if not os.path.exists(data_path):
    os.mkdir(data_path)  
if not os.path.exists(os.path.join(data_path, 'plots')):
    os.mkdir(os.path.join(data_path, 'plots'))
if not os.path.exists(os.path.join(data_path, 'plots_label')):
    os.mkdir(os.path.join(data_path, 'plots_label'))

# label_folder = '05b_labels'
# label_folder = '05b_labels_dry_seg_iso'
label_folder = '05b_labels_from_5'
#label_folder = '05b_labels_dry_seg_iso'

# transition_folder = '03_gradient_filtered_transitions'
# transition_2_folder = '03_gradient_filtered_transitions2'

transition_folder = '03_gradient_filtered_transitions_from_5'
transition_2_folder = '03_gradient_filtered_transitions2_from_5'

#transition_folder = '03_b_gradient_filtered_transitions_enhanced'
#transition_2_folder = '03_b_gradient_filtered_transitions2_enhanced5'

good_samples = robpylib.TOMCAT.INFO.good_samples
#good_samples = robpylib.TOMCAT.INFO.samples_to_repeat

#time_0 = time.time()

def fit_fun(data, start, end, time, n=1):
    start = np.uint16(start)
    end = np.uint16(end)
    if len(data[start:end])<1:
        return np.poly1d([ 0, 0])
    if n==1:
        lin_fit = np.polyfit(time[start:end], data[start:end], 1)
        fitted = np.poly1d(lin_fit)
        return fitted


def dyn_fit(sample_data, data_path= data_path):
#    sample_data = xr.open_dataset(os.path.join(data_path, sample))
#    total_volume = sample_data['volume'].sum(axis = 0)

#TO DO load for all pores volume(t), start, end and then fit linear-law, sqrt-law or pwr-law fk=Ak*(t-t0k)**nk
# then total_volume_fit(t) = sum (max(0, min(fk(t), pore_volume(end))))
# store Ak, t0k, nk and try to compare to pore props

    pore_volume = sample_data['volume']
    time = sample_data['time'].data
    labels = np.uint16(sample_data['label'].data)
    num_pores = len(labels)
    fits = []

    for label in range(num_pores):
        start = np.uint16(sample_data['dynamics'][label,0].data)
        end = np.uint16(sample_data['dynamics'][label,1].data)
        data = pore_volume[label].data
        dat_fit= fit_fun(data, start, end, time)
        fits.append(dat_fit)
    
    fit_data = np.zeros([len(fits), 2])
    
    c=0
    for fit in fits:
       fit_data[c, 0] = fit[1]   #slope
       fit_data[c, 1] = -fit[0]/fit[1]  #onset
       if fit_data[c, 1] < 0: fit_data[c, 1] = 0
       c = c+1
    
    fit_data[np.isnan(fit_data)]=0
    return fit_data


def get_Dyn_Data(sample, baseFolder=baseFolder):

#load data    
    tension = np.uint16(sample[3:6])
    sample_id = np.uint8(sample[7])
    last_3 = sample[-3:]
    if last_3 == 'III':
        repeat = 3
    elif last_3 == '_II':
        repeat = 2
    else:
        repeat = 1
    
    transitionFolder = os.path.join(baseFolder, sample, transition_folder)
    transition_2_Folder = os.path.join(baseFolder,sample, transition_2_folder)
    labelFolder = os.path.join(baseFolder, sample, label_folder) 
    
    transitions, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(transitionFolder, filetype=np.uint16, track=False)
    labels, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(labelFolder, track=False)
    
    if sample[0]=='T':
        transitions2, _ = robpylib.CommonFunctions.ImportExport.ReadStackNew(transition_2_Folder, filetype=np.uint8, track=False)
    else: transitions2 = np.zeros(transitions.shape, dtype=np.uint8)
    
#    crop volume to ROI clearly above the water line
    transitions = transitions[:,:,:waterline]
    labels = labels[:,:,:waterline]
    transitions2 = transitions2[:,:,:waterline]
    
    if sample in robpylib.TOMCAT.INFO.samples_to_repeat:
        transitions[transitions>0] = transitions[transitions>0]+5
        transitions2[transitions2>0] = transitions2[transitions2>0]+5
    
    limit = 400
    
    if sample in list(time_limit.keys()):
        limit = time_limit[sample]
    
    transitions[transitions>limit -1 ]=0       
    label_id, label_count = np.unique(labels,return_counts=True)
  
    relevant_pores = np.where(label_count>50)[0][1:]   #remove value 0 from array (= background)
    number_pores = len(relevant_pores)
    relevant_labels = label_id[relevant_pores]

# first step: extract dynamics  
    
    t_max = transitions.max()+1
    
    if sample[0]=='T':
        time_list = np.array(robpylib.TOMCAT.TIME.TIME[sample])
        t_max = time_list.shape[0]
    else:
        time_list=np.arange(t_max)*15 #[s]
    step_size = time_list.copy()
    step_size[1:] = np.diff(time_list)
    
#    if sample in robpylib.TOMCAT.INFO.samples_to_repeat:
#        time_list = time_list[5:]
#        step_size = step_size[5:]
    
    
    data_filling = np.zeros([number_pores, t_max], dtype=np.uint32)
    
    pore=0
    
    labelmask = np.zeros(labels.shape, dtype=np.bool) 

#    print('number of relevant labels')
#    print(len(relevant_labels))
    for label in relevant_labels:
        mask = labels==label
        labelmask[mask] = True
        mask = np.uint8(mask)
        TEST = transitions*mask    
        time_step, time_filling = np.unique(TEST[np.where(TEST>0)],return_counts=True)
        TEST = transitions2*mask
        back_step, back_filling = np.unique(TEST[np.where(TEST>0)],return_counts=True)
        
        pore = pore+1
        if len(time_step)>0:
            data_filling[pore-1, time_step] = time_filling
            
            if len(back_step)>0:
                data_filling[pore-1, back_step] = data_filling[pore-1, back_step] - back_filling
    data_volume = np.cumsum(data_filling, axis=1).astype(np.uint32)
    
#    print('time pore analysis')
#    time_pore = time.time()
#    print(time_pore-time_0)
    
    filling_scaled = data_filling/ step_size[None,:]
    filling_scaled[np.isnan(filling_scaled)] = 0   
    
# second step: analyse dynamics

    dynamics= np.zeros([number_pores,10])
    
    for pore in range(number_pores):
        maxval = data_volume[pore,:].max()
        if maxval >0:
            onset = np.where(data_volume[pore,:]>0.05*maxval)[0][0]
            end = np.where(data_volume[pore,:]>0.99*maxval)[0][0]
            dynamics[pore, :2] = [onset, end]
            dynamics[pore, 2:4] = [time_list[onset], time_list[end]]
            
            peaks, props = sp.signal.find_peaks(data_filling[pore,:], height=500)
            dynamics[pore, 7] = len(peaks)
            
        pore=pore+1  
    
    dynamics[:, 4] = np.max(filling_scaled, axis = 1)
    dynamics[:, 5] = np.argmax(filling_scaled, axis = 1)
    dynamics[:, 6] = np.median(filling_scaled[:,onset:end], axis = 1)
    dynamics[:, 8] = data_volume[:,-1]
    dynamics[:, 9] = data_volume[:,-1]/label_count[relevant_pores]
    
#    print('interval detection time analysis')
#    print(time.time()-time_pore)

# step 3: store to xarray dataset
    parameter = ['filling start time step', 'filling end time step','filling start [s]', 'filling end [s]', 'max filling rate [vx_s]', 'time step of max filling rate', 'median filling rate [vx_s]', 'num filling peaks', 'final water volume [vx]', 'final_saturation']    
    
    
    sample_data = xr.Dataset({'filling': (['label', 'time'], filling_scaled),
                              'volume': (['label', 'time'], data_volume),
                              'dynamics': (['label', 'parameter'], dynamics),
                              'transition_matrix': (['x','y','z'], transitions),
                              'transition_2_matrix': (['x','y','z'], transitions2),
                              'label_matrix': (['x','y','z'], labels)},
                              coords = {'label': relevant_labels,
                                      'time': time_list,

                                      'parameter': parameter,
                                      'x': np.arange(labels.shape[0]),
                                      'y': np.arange(labels.shape[1]),
                                      'z': np.arange(labels.shape[2])},
                                      attrs = {'name': sample,
                                               'sample_id': sample_id,
                                               'tension': tension,
                                               'repeat': repeat,
                                               'voxel_size': '2.75 um',
                                               'voxel': 2.75E-6, #m
                                               'comment': 'units in voxel, seconds or voxel/second; time steps in transition matrix',
                                               'label folder': label_folder,
                                               'transition_folder': transition_folder,
                                               'transition_2_folder': transition_2_folder,
                                               'waterline = ROI[:,:,:waterline]': waterline})
    sample_data['filling'].attrs['units'] = 'voxel/s'
    sample_data['volume'].attrs['units'] = 'voxel'
    sample_data['transition_matrix'].attrs['units'] = 'time step'
    sample_data['transition_2_matrix'].attrs['units'] = 'time step'
    sample_data['label_matrix'].attrs['units'] = 'pore label'
    sample_data['time'].attrs['units'] = 's'
    sample_data['x'].attrs['units'] = 'voxel'
    sample_data['y'].attrs['units'] = 'voxel'
    sample_data['z'].attrs['units'] = 'voxel'
    
    fit_data = dyn_fit(sample_data)
    
    sample_data.coords['fit_var'] = ['slope', 'onset'] 
    sample_data['fit_data'] = (['label', 'fit_var'], fit_data)
    
    
# store each sample Dataset in separate files, better: find way to include 
# sample and sample parameter as dimension in single dataset, easier: just loop over all sample files if post-processing is necessary
    return sample_data  # return sample_array and put it in datase


def sigmoid_fun(x, x0, beta, alpha):    
    y = alpha/(1+np.exp(-beta*(x-x0)))
    return y

def R_squared(x,y, func, *p):
    R = 0
    y_mean = np.mean(y)
    if y_mean > 0:
        SStot = np.sum((y-y_mean)**2)
        SSres = np.sum((y-func(x,*p))**2)
        R = 1 - SSres/SStot
    return R

def dyn_fit_step2(sample_data):
    total_volume = sample_data['volume'].sum(axis = 0)
    pore_volume = sample_data['volume']
    time = sample_data['time'].data
    labels = np.uint16(sample_data['label'].data)
    num_pores = len(labels)
    sample_name = sample_data.attrs['name']
    t_max = -1
    
    if sample_name in list(time_limit.keys()):
        t_max = time_limit[sample_name]
    
    fits = []
    fits_sig = np.zeros([num_pores,4])

    for label in range(num_pores):
        
        color = labels[label]
        fit_data = sample_data['fit_data'][label, :]
        par_0 = -fit_data[1]*fit_data[0]
        if np.isnan(par_0): par_0 = 0               #reconversion to the adeqaute format of poly1d
        par_1 = fit_data[0]
        fit_fun = np.poly1d([par_1, par_0])
        fits.append(fit_fun)
        
        start, end = sample_data['dynamics'][label, 2:4]
        center = start+(end-start)/2
        
        mass_data = pore_volume[label,:].data
        
        try:
            p_sig, cov_sig = scipy.optimize.curve_fit(sigmoid_fun, time, mass_data, maxfev = 200000, p0=[center, 0.1, mass_data[-1]])
            R_sig = R_squared(time, mass_data, sigmoid_fun, *p_sig)
        except Exception:
            p_sig = [0, 0, 0]
            R_sig = 0
        
        fits_sig[label,:3] = p_sig
        fits_sig[label,3] = R_sig
        
        
        plt.figure()
        plt.plot(time[:t_max], mass_data[:t_max], 'k')
        plt.plot(time[:t_max], sigmoid_fun(time[:t_max], *p_sig), 'r')
        plt.title(color)
        plt.xlabel('time [s]')
        plt.ylabel('volume [vx]')
        filename = os.path.join(data_path, 'plots_label', ''.join([sample_data.attrs['name'],'_label_',str(color),'.png']))
        plt.savefig(filename, dpi=500, bbox_inches = 'tight')    
    
        
    fit_volume = np.zeros(len(time))
    crude_fit_volume = np.zeros(len(time))
    sig_fit_volume = np.zeros(len(time))
    
    for ts in range(len(time)):
        t = time[ts]
        val = 0
        val2 = 0
        val3 = 0
        for i in range(len(fits)):
            end = np.uint16(sample_data['dynamics'][i,1].data)
            start = np.uint16(sample_data['dynamics'][i,0].data)
#            val = val + np.max([0, np.min([fits[i](t)-fits[i](0), pore_volume[i][end]])])
            val = val + np.max([0, np.min([fits[i](t), pore_volume[i][end]])])
            
            if ts > start:
                dval2 = np.min([pore_volume[i][end]/(end-start)*(ts-start), pore_volume[i][end]])
                if np.isnan(dval2): dval2=0
                val2 = val2 + dval2
            p_sig = fits_sig[i,:3]   
            
            val3 = val3 + sigmoid_fun(time[ts], *p_sig)
            
        fit_volume[ts] = val
        crude_fit_volume[ts] = val2
        sig_fit_volume[ts] = val3
    
#    
    plt.figure()
    total_volume[:t_max].plot(color='k')
#    plt.plot(time, fit_volume)
#    plt.plot(time, crude_fit_volume)
    plt.plot(time[:t_max], sig_fit_volume[:t_max], 'r')
    plt.title(sample_data.attrs['name'])
    filename = os.path.join(data_path, 'plots', ''.join([sample_data.attrs['name'],'.png']))

    plt.savefig(filename, dpi=500, bbox_inches = 'tight')
    fits_sig = np.array(fits_sig)
    
    sample_data.coords['sig_fit_var'] = ['t0 [s]', 'beta [1_s]', 'alpha [vx]', 'R2']
    sample_data['sig_fit_data'] = (['label', 'sig_fit_var'], fits_sig)
    sample_data.attrs['sigmoidal_fit'] = 'm(t) = alpha/(1+np.exp(-beta*(t-ts0)), units voxel and seconds'
    
    
    return sample_data
c=0
samples_crude = os.listdir(baseFolder)
#samples = ['32_200_025H2_cont']
samples = os.listdir(baseFolder)
# samples = []
# for sample in samples_crude:
#     if sample in robpylib.TOMCAT.INFO.samples_to_repeat:
#         samples.append(sample)

def mainfunction(sample, baseFolder = baseFolder, data_path = data_path):
    if not sample == 'T3_025_1':
        if os.path.exists(os.path.join(baseFolder, sample, label_folder)):
            name = ''.join(['dyn_data_',sample,'.nc'])
            filename = os.path.join(data_path, name)
            if not os.path.exists(filename):
                sample_data = get_Dyn_Data(sample)
                sample_data = dyn_fit_step2(sample_data)
                sample_data.to_netcdf(filename)
                return sample_data

# if not parallel:  
#     for sample in samples:
#         c= c+1
#     #    if c>1: continue
#     #    if sample[1]=='4': continue
#         if not os.path.exists(os.path.join(baseFolder, sample, label_folder)): continue
#     #    if not sample == '32_200_025H2_cont': continue
#     #    if not sample == 'T3_025_3_III': continue
#         print(sample, ''.join(['(',str(c),'/',str(len(samples)),')']))
#         name = ''.join(['dyn_data_',sample,'.nc'])
#         filename = os.path.join(data_path, name)
#         if os.path.exists(filename):
#             print('already processed, skipping ...')
#             continue
#         sample_data = get_Dyn_Data(sample)
#         sample_data = dyn_fit_step2(sample_data)
#         sample_data.to_netcdf(filename)
    
if parallel:
    results=Parallel(n_jobs=num_cores)(delayed(mainfunction)(sample) for sample in samples)
# for sample in samples:
#     print(sample)
#     mainfunction(sample)
    
        
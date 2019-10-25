# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:58:26 2019

@author: firo
"""

import sys
library=r"R:\Scratch\305\_Robert\Python_Library"

if library not in sys.path:
    sys.path.append(library)
    
    
import xarray as xr
import os
#import scipy as sp
import scipy.optimize
#import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

drive = '//152.88.86.87/data118'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcfd4_archives', 'processed')



def sigmoid_fun(x, x0, beta, alpha):    #fix gamma to pore volume?!
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
    
    


def dyn_fit_step2(sample, data_path= data_path):
    sample_data = xr.open_dataset(os.path.join(data_path,'temp', sample))
    total_volume = sample_data['volume'].sum(axis = 0)

#TO DO load for all pores volume(t), start, end and then fit linear-law, sqrt-law or pwr-law fk=Ak*(t-t0k)**nk
# then total_volume_fit(t) = sum (max(0, min(fk(t), pore_volume(end))))
# store Ak, t0k, nk and try to compare to pore props

    pore_volume = sample_data['volume']
    time = sample_data['time'].data
    labels = np.uint16(sample_data['label'].data)
    num_pores = len(labels)
   
    
    fits = []
    fits_sig = []

    for label in range(num_pores):
        
        fit_data = sample_data['fit_data'][label, :]
        par_0 = -fit_data[1]*fit_data[0]
        if np.isnan(par_0): par_0 = 0               #reconversion to the adeqaute format of poly1d
        par_1 = fit_data[0]
        fit_fun = np.poly1d([par_1, par_0])
        fits.append(fit_fun)
        
        start, end = sample_data['dynamics'][label, :2]
        center = start+(end-start)/2
        
        mass_data = pore_volume[label,:].data
        
        try:
            p_sig, cov_sig = scipy.optimize.curve_fit(sigmoid_fun, time, mass_data, maxfev = 200000, p0=[center, 0.1, mass_data[-1]])
            R_sig = R_squared(time, mass_data, sigmoid_fun, *p_sig)
            np.concatenate(p_sig, R_sig)
        except Exception:
            p_sig = [0, 0, 0, 0]
        fits_sig.append(p_sig)
        plt.figure()
        plt.plot(time, mass_data)
        plt.plot(time, sigmoid_fun(time, *p_sig))
        plt.title(label)
        filename = os.path.join(data_path, 'plots_label', ''.join([sample_data.attrs['name'],'_label_',str(label),'.png']))
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
            p_sig = fits_sig[i][:-1]    
            
            val3 = val3 + sigmoid_fun(time[ts], *p_sig)
            
        fit_volume[ts] = val
        crude_fit_volume[ts] = val2
        sig_fit_volume[ts] = val3
    
#    
    plt.figure()
    total_volume.plot()
    plt.plot(time, fit_volume)
    plt.plot(time, crude_fit_volume)
    plt.plot(time, sig_fit_volume)
    plt.title(sample_data.attrs['name'])
    filename = os.path.join(data_path, 'plots', ''.join([sample_data.attrs['name'],'.png']))

    plt.savefig(filename, dpi=500, bbox_inches = 'tight')
    fits_sig = np.array(fits_sig)
    
    sample_data.coords['sig_fit_var'] = ['t0', 'beta', 'alpha', 'R2']
    sample_data['sig_fit_data'] = (['label', 'sig_fit_var'], fits_sig)
    sample_data.attrs['sigmoidal_fit'] = 'm(t) = alpha/(1+np.exp(-beta*(t-ts0)), units voxel and seconds'
    sample_data.to_netcdf(os.path.join(data_path, sample))
    
    
    return True#fit_volume, crude_fit_volume, sig_fit_volume

# compare fit parameters to pore size etc., maybe there is a correlation
samples = os.listdir(data_path)
#samples = [samples[2]]

for sample in samples:
    print(sample)
#    if not sample == 'dyn_data_T3_025_3_III.nc': continue
    if not sample[:3] == 'dyn': continue
    if sample == 'temp': continue
    if sample == 'test': continue
    if sample == 'plots': continue
    if sample == 'plots_temp': continue
#    fit_volume, crude_fit_volume, sig_fit_volume = dyn_fit(sample)
#    test = dyn_fit_step2(sample)

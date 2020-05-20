# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:32:52 2020

@author: firo
"""

import xarray as xr
import numpy as np
import scipy as sp
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.optimize


drive = '//152.88.86.87/data118'
processing_version = 'processed_1200_dry_seg_aniso_sep'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives', processing_version)

samples = os.listdir(data_path)

total_sample = 'total_energy_data_v3_1_T3_025_3_III.nc'
pore_sample = 'energy_data_v3_4_T3_025_3_III.nc'
dyn_sample = 'dyn_data_T3_025_3_III.nc'

pxm = 2.75E-6
axm2 = pxm**2
vxm3 = pxm**3
gamma = 72E-3 #N/m
theta = 50 #°
cos = np.cos(theta/180*np.pi)


def resample_data(data, time):
    spline = interp1d(time, data, fill_value = 'extrapolate')
    new_time = np.arange(time.min(),time.max(),1)
    new_data = spline(new_time)
    return new_time, new_data

def diff_free_energy(dA_wa, dA_ws, gamma=gamma, cos=cos):
    dF = gamma*(dA_wa - dA_ws*cos)
    return dF

def free_energy(A_wa, A_ws, gamma=gamma, cos=cos):
    F = gamma*(A_wa-cos*A_ws)
    return F

def time_deriv_1d(data, time):
    dD = np.gradient(data, time)
    return dD

def calc_interface(dataset, axm2=axm2):
    interface = dataset['total_water_surface']-dataset['water_solid_area_by_label']-dataset['water_water_area_by_label']
    interface = interface*axm2
    return interface

def calc_wet_surface(dataset, axm2=axm2):
    wet_surface = axm2*dataset['water_solid_area']
    return wet_surface

def squared(x, k):
    y=k*x**2
    return y


pore_scale_data = xr.load_dataset(os.path.join(data_path, pore_sample))
dyn_data = xr.load_dataset(os.path.join(data_path, dyn_sample))

time = dyn_data['time'].data
flux = dyn_data['filling'].data
A_ws = calc_wet_surface(pore_scale_data)
A_wa = calc_interface(pore_scale_data)

F = free_energy(A_wa, A_ws).data
dF = np.gradient(F, time, axis=1)

flux = flux[dF>-2E-10]
dF = dF[dF>-2E-10]
flux = flux[dF<0.5E-10]
dF = dF[dF<0.5E-10]

hist = np.histogram(dF, bins=100, density = True)
hist_w = np.histogram(dF, weights=flux,bins=100, density=True)

plt.plot(hist[1][1:], hist[0], label='raw')
plt.plot(hist_w[1][1:], hist_w[0], label='volume flux weighted')
plt.legend()
plt.title(dyn_data.attrs['name'])
plt.xlabel('energy flux [J/s]')
plt.ylabel('frequency [arb. units]')
plt.ylim(-0.25E10,0.5E11)

filename = r"H:\03_Besprechungen\Group Meetings\May_2020\energy_histogram_no_crop_zoom.png"
# plt.savefig(filename, bbox_inches = 'tight', dpi=600)

flux=flux*vxm3
R= -2*dF/flux**2
plt.figure()
plt.yscale('log')
plt.plot(dF[dF<0], R[dF<0], '.')
plt.plot(dF[dF>0], -R[dF>0], '.')
plt.xlabel('energy flux [J/s]')
plt.ylabel('resistance [Pas/m3]')
plt.title('median = 7.27E18 Pas/m3')
filename = r"H:\03_Besprechungen\Group Meetings\May_2020\resistance.png"
# plt.savefig(filename, bbox_inches = 'tight', dpi=600)

plt.figure()
plt.yscale('log')
plt.plot(flux[dF<0], R[dF<0], '.')
plt.plot(flux[dF>0], -R[dF>0], '.')
plt.xlabel('Q^2 [m3/s]^2')
plt.ylabel('resistance [Pas/m3]')
plt.title('median = 7.27E18 Pas/m3')
filename = r"H:\03_Besprechungen\Group Meetings\May_2020\resistance_over_flux.png"
# plt.savefig(filename, bbox_inches = 'tight', dpi=600)

plt.figure()
plt.plot(flux, dF ,'.')
# plt.plot(flux_test, lin_fun(flux_test, *p_lin), 'k--')
filename = r"H:\03_Besprechungen\Group Meetings\May_2020\capillary_pressure.png"
plt.xlabel('volume flux [m3/s]')
plt.ylabel('energy flux [J/s]')
# plt.savefig(filename, bbox_inches = 'tight', dpi=600)

plt.figure()
plt.plot(-dF*2/flux**2, flux,  '.')
plt.xlim(-0.5E19,0.5E19)
plt.xlabel('resistance [Pas/m3]')
plt.ylabel('flux [m3/s]')
filename = r"H:\03_Besprechungen\Group Meetings\May_2020\resistance_dist.png"
# plt.savefig(filename, bbox_inches = 'tight', dpi=600)

plt.figure()
plt.yscale('log')
plt.plot(flux[dF<0], R[dF<0], '.')
plt.plot(flux[dF>0], -R[dF>0], '.')
# plt.plot(flux_test, R0+1/(1E8*flux_test)**3,'k')
# plt.plot(flux_test, R0+a/(b*flux_test)**2,'r')
plt.ylabel('resistance [Pas/m3]')
plt.xlabel('flux [m3/s]')
filename = r"H:\03_Besprechungen\Group Meetings\May_2020\resistance_error_guess.png"
# plt.savefig(filename, bbox_inches = 'tight', dpi=600)


# p, cov = sp.optimize.curve_fit(squared, dF*flux, flux, maxfev = 50000, p0=[-2300])
p = [-2550]
plt.plot(flux, dF*flux, '.')
plt.xlim(0,5.5E-14)
plt.ylim( -1E-23,0.1E-23)
plt.plot(np.arange(0,5.5E-14,0.1E-14), squared(np.arange(0,5.5E-14,0.1E-14), *p))
plt.xlabel('volume flux [m3/s]')
plt.ylabel('volume flux*energy flux [Pa m6/s2]')
filename = r"H:\03_Besprechungen\Group Meetings\May_2020\mean_pressure.png"
plt.savefig(filename, bbox_inches = 'tight', dpi=600)

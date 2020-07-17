# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage

data = xr.load_dataset(r"A:\Robert_TOMCAT_4_netcdf4\dyn_data_T4_025_1_III.nc") 
data_en = xr.load_dataset(r"A:\Robert_TOMCAT_4_netcdf4\total_energy_data_v3_1_T4_025_1_III.nc")


pxm = 2.75E-6
axm2 = pxm**2
vxm3 = pxm**3

gamma = 72E-3 #N/m
cos = np.cos(48/180*np.pi)

pxmm = 2.75E-3
axmm2 = pxmm**2
vxmm3 = pxmm**3

sig = 0.75
V = data['volume'].data
time = data['time'].data

V = V.sum(axis=0)

V = sp.ndimage.filters.gaussian_filter1d(V, sig)*vxm3

dV = np.gradient(V, time)[1:]


inttime = data_en['time'].data

A_wa = data_en['interfaces'].sel(area='A_wa_corr')
A_wa_smooth = A_wa.copy()
A_wa_smooth = sp.ndimage.filters.gaussian_filter1d(A_wa_smooth, sig)
dA_wa = np.gradient(A_wa_smooth, inttime)

A_ws = data_en['interfaces'].sel(area='A_ws')
A_ws_smooth = A_ws.copy()
A_ws_smooth = sp.ndimage.filters.gaussian_filter1d(A_ws_smooth, sig)
dA_ws = np.gradient(A_ws_smooth, inttime)

F = gamma*axm2*(A_wa_smooth - cos*A_ws_smooth)
dF = gamma*axm2*(dA_wa - cos*dA_ws)

R = -dF/dV**2

plt.plot(inttime, R)

plt.figure()
# plt.plot(dV[:220], dV[:220]*dF[:220], 'k')
plt.scatter(dV[:220], dV[:220]*dF[:220], c=inttime[:220], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('flux*energy flux')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1E-17,1E-11)
plt.ylim(1E-31,1E-21)

plt.figure()
# plt.plot(dV[:220], -dF[:220]/dV[:220], 'k')
plt.scatter(dV[:220], -dF[:220]/dV[:220], c=inttime[:220], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('capillary pressure [Pa]')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(1E-17,1E-11)
plt.ylim(-2500,5000)

lim=5E15
plt.figure()
# plt.plot(dV[:220], -2*dF[:220]/dV[:220]**2, 'k')
plt.scatter(dV[:220], -2*dF[:220]/dV[:220]**2, c=inttime[:220], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('resistance [Pa s/m3]')
plt.ylim(-lim/2, lim)

# plt.figure()
# plt.plot(inttime[:220], -2*dF[:220]/dV[:220]**2,'.' )
# plt.ylim(-lim/2, lim)
# [1:221]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
ax1.scatter(inttime[:220], np.abs(-2*dF[:220]/dV[1:221]**2), c= dV[1:221],cmap='plasma', marker='x')
ax2.scatter(inttime[:220], np.abs(-dF[0:220]/dV[:220]), c= dV[:220],cmap='plasma')
ax3.plot(inttime[:220], V[1:221], 'k')
ax4.plot(inttime[:220], dV[:220], 'r')
ax1.set_ylim(0, lim)
ax2.set_ylim(0, 5000)




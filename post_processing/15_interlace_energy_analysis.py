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

# data = xr.load_dataset(r"A:\Robert_TOMCAT_4_netcdf4\dyn_data_T4_025_1_III.nc") 
# data_en = xr.load_dataset(r"A:\Robert_TOMCAT_4_netcdf4\total_energy_data_v3_1_T4_025_1_III.nc")

data = xr.load_dataset(r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep\dyn_data_T3_025_3_III.nc")
data_en = xr.load_dataset(r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep\total_energy_data_v3_1_T3_025_3_III.nc")

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

V = V.sum(axis=0)*vxm3

# V = sp.ndimage.filters.gaussian_filter1d(V, sig)*vxm3

dV = np.gradient(V, time)[1:]
DV = np.gradient(V)


inttime = data_en['time'].data

A_wa = data_en['interfaces'].sel(area='A_wa_corr')
A_wa_smooth = A_wa.copy()
# A_wa_smooth = sp.ndimage.filters.gaussian_filter1d(A_wa_smooth, sig)
dA_wa = np.gradient(A_wa_smooth, inttime)
DA_wa = np.gradient(A_wa_smooth)

A_ws = data_en['interfaces'].sel(area='A_ws')
A_ws_smooth = A_ws.copy()
# A_ws_smooth = sp.ndimage.filters.gaussian_filter1d(A_ws_smooth, sig)
dA_ws = np.gradient(A_ws_smooth, inttime)
DA_ws = np.gradient(A_ws_smooth)

F = gamma*axm2*(A_wa_smooth - cos*A_ws_smooth)
dF = gamma*axm2*(dA_wa - cos*dA_ws)
DF = gamma*axm2*(DA_wa - cos*DA_ws)


R = -dF/dV**2

plt.plot(inttime, R)
tt=358
plt.figure()
# plt.plot(dV[:220], dV[:220]*dF[:220], 'k')
plt.scatter(dV[:tt], dV[:tt]*dF[:tt], c=inttime[:tt], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('flux*energy flux')
plt.xscale('log')
plt.yscale('log')
# plt.xlim(1E-17,1E-11)
# plt.ylim(1E-31,1E-21)

plt.figure()
# plt.plot(dV[:220], -dF[:220]/dV[:220], 'k')
plt.scatter(dV[:tt], -np.gradient(F[:tt], V[1:tt+1]), c=inttime[:tt], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('capillary pressure [Pa]')
#plt.xscale('log')
#plt.yscale('log')
# plt.xlim(1E-17,1E-11)
plt.ylim(-2500,5000)

lim=5E15
plt.figure()
# plt.plot(dV[:220], -2*dF[:220]/dV[:220]**2, 'k')
plt.scatter(dV[:tt], -2*dF[:tt]/dV[:tt]**2, c=inttime[:tt], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('resistance [Pa s/m3]')
plt.ylim(-lim/2, lim)
qlim = dV.max()
# plt.figure()
# plt.plot(inttime[:220], -2*dF[:220]/dV[:220]**2,'.' )
# plt.ylim(-lim/2, lim)
# [1:221]


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
ax5 = ax1.twinx()
R= np.abs(-dF[:tt]/dV[:tt]**2)
# p = np.abs(-DF[0:tt]/DV[:tt])
p = np.abs(np.gradient(F[:tt], V[1:tt+1]))
rat = p/R
t0=0
rat[~np.isfinite(rat)] = 0
ax5.plot(inttime[t0:tt], rat[t0:],'--')
# ax1.scatter(inttime[:220], R, c= dV[1:221],cmap='plasma', marker='x')
# ax2.scatter(inttime[:220],p , c= dV[:220],cmap='plasma')
ax3.plot(inttime[:tt], V[:tt], 'k')
ax4.plot(inttime[:tt], dV[:tt], 'r')
ax1.set_ylim(0, lim)
ax2.set_ylim(0, 5000)
ax5.set_ylim(0,qlim)
ax4.set_ylim(0,qlim)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(inttime[:tt], V[:tt], 'k')
ax1.plot(inttime[:tt], np.log10(p*R), '.')



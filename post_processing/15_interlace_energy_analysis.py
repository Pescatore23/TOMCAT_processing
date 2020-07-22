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
import os

data = xr.load_dataset(r"A:\Robert_TOMCAT_4_netcdf4\dyn_data_T4_025_1_III.nc") 
data_en = xr.load_dataset(r"A:\Robert_TOMCAT_4_netcdf4\total_energy_data_v3_1_T4_025_1_III.nc")

def weighted_ecdf(data, weight = False):
    """
    input: 1D arrays of data and corresponding weights
    sets weight to 1 if no weights given (= "normal" ecdf, but better than the statsmodels version)
    """
    if not np.any(weight):
        weight = np.ones(len(data))

    sorting = np.argsort(data)
    x = data[sorting]
    weight = weight[sorting]
    y = np.cumsum(weight)/weight.sum()

    # clean duplicates, statsmodels does not do this, but it is necessary for us

    x_clean = np.unique(x)
    y_clean = np.zeros(x_clean.shape)

    for i in range(len(x_clean)):
        y_clean[i] = y[x==x_clean[i]].max()
    return x_clean, y_clean

# data = xr.load_dataset(r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep\dyn_data_T3_025_3_III.nc")
# data_en = xr.load_dataset(r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep\total_energy_data_v3_1_T3_025_3_III.nc")
plot_path = os.path.join(r"A:\Robert_TOMCAT_4_netcdf4", 'plots', data.attrs['name'])
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

pxm = 2.75E-6
axm2 = pxm**2
vxm3 = pxm**3

gamma = 72E-3 #N/m
cos = np.cos(48/180*np.pi)

pxmm = 2.75E-3
axmm2 = pxmm**2
vxmm3 = pxmm**3

sig = 0.5
V = data['volume'].data
time = data['time'].data

V = V.sum(axis=0)

V = sp.ndimage.filters.gaussian_filter1d(V, sig)*vxm3

dV = np.gradient(V, time)[1:]
DV = np.gradient(V)


inttime = data_en['time'].data

A_wa = data_en['interfaces'].sel(area='A_wa_corr')
A_wa_smooth = A_wa.copy()
A_wa_smooth = sp.ndimage.filters.gaussian_filter1d(A_wa_smooth, sig)
dA_wa = np.gradient(A_wa_smooth, inttime)
DA_wa = np.gradient(A_wa_smooth)

A_ws = data_en['interfaces'].sel(area='A_ws')
A_ws_smooth = A_ws.copy()
A_ws_smooth = sp.ndimage.filters.gaussian_filter1d(A_ws_smooth, sig)
dA_ws = np.gradient(A_ws_smooth, inttime)
DA_ws = np.gradient(A_ws_smooth)

F = gamma*axm2*(A_wa_smooth - cos*A_ws_smooth)
dF = gamma*axm2*(dA_wa - cos*dA_ws)
DF = gamma*axm2*(DA_wa - cos*DA_ws)



tt=220
plt.figure()
# plt.plot(dV[:220], dV[:220]*dF[:220], 'k')
plt.scatter(dV[:tt], -dV[:tt]*dF[:tt], c=inttime[:tt], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('flux*energy flux')
plt.xscale('log')
plt.yscale('log')
plt.xlim(8E-14,0.5E-10)
plt.ylim(1E-24,0.5E-17)
filename = os.path.join(plot_path, 'flux_times_energy.png')
plt.savefig(filename, dpi=500, bbox_inches = 'tight')

plt.figure()
# plt.plot(dV[:220], -dF[:220]/dV[:220], 'k')
plt.scatter(dV[:tt], -np.gradient(F[:tt], V[1:tt+1]), c=inttime[:tt], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('capillary pressure [Pa]')
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(5E-14,2.53E-11)
plt.ylim(-2500,5000)
filename = os.path.join(plot_path, 'capillary_pressure.png')
plt.savefig(filename, dpi=500, bbox_inches = 'tight')

lim=5E15
plt.figure()
# plt.plot(dV[:220], -2*dF[:220]/dV[:220]**2, 'k')
plt.scatter(dV[:tt], -dF[:tt]/dV[:tt]**2, c=inttime[:tt], cmap='viridis', zorder=10)
plt.colorbar(label = 'time [s]')
plt.xlabel('flux [m3/s]')
plt.ylabel('resistance [Pa s/m3]')
plt.ylim(-lim/2, lim)
qlim = dV.max()
# plt.figure()
# plt.plot(inttime[:220], -2*dF[:220]/dV[:220]**2,'.' )
# plt.ylim(-lim/2, lim)
# [1:221]
filename = os.path.join(plot_path, 'resistance.png')
plt.savefig(filename, dpi=500, bbox_inches = 'tight')


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
ax5 = ax1.twinx()
ax6 = ax1.twinx()
R= np.abs(-dF[:tt]/dV[:tt]**2)
p = np.abs(-dF[0:tt]/dV[1:tt+1])
# p = -np.gradient(F[:tt], V[1:tt+1])
rat = p/R
t0=0
rat[~np.isfinite(rat)] = 0
ax5.plot(inttime[t0:tt], rat[t0:],'--')
ax5.axes.yaxis.set_ticks([])
qlim=qlim*1.1
ax1.scatter(inttime[:220], R, c= dV[1:221],cmap='plasma', marker='x')
ax1.set_yscale('log')
ax1.set_ylabel('resistance [Pa s/m3')
# ax1.scatter(inttime[:tt], np.sqrt(-dF[:tt]*dV[:tt]) , c= dV[:220],cmap='plasma')
im = ax2.scatter(inttime[:220],p , c= dV[:220],cmap='plasma')
fig.colorbar(im, label='flux [m3/s]',  orientation='vertical', pad=0.2)
ax2.set_ylabel('capillary pressure [Pa]')
ax3.plot(inttime[:tt], V[:tt], 'k')
ax3.axes.yaxis.set_ticks([])

ax4.plot(inttime[:tt], dV[:tt], 'r')
ax4.axes.yaxis.set_ticks([])

ax1.set_ylim(1E14, 5*lim)
# ax1.set_ylim(0, 1.4E-18)
ax2.set_ylim(-500, 5000)
ax5.set_ylim(0,qlim)
ax4.set_ylim(0,qlim)
ax1.set_xlabel('time [s]')
dV_raw = np.gradient(data['volume'].sum(axis=0), data['time'])*vxm3
ax6.plot(time[1:tt+1], dV_raw[1:tt+1], 'k-.')
ax6.axes.yaxis.set_ticks([])
ax6.set_ylim(0,qlim)
filename = os.path.join(plot_path, 'res_p_over_time.png')
plt.savefig(filename, dpi=500, bbox_inches = 'tight')


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax2.plot(inttime[:tt], V[:tt], 'k')
ax1.plot(inttime[:tt], (p*R), '.')
ax1.set_yscale('log')
ax1.set_xlabel('time [s]')
ax2.set_ylabel('volume [m3]')
ax1.set_ylabel('p*R')
ax1.set_ylim(1E16,1E20)
filename = os.path.join(plot_path, 'p_times_r.png')
plt.savefig(filename, dpi=500, bbox_inches = 'tight')
const = np.abs(p*R)
const = const[np.isfinite(const)]
x,y = weighted_ecdf(const)

plt.figure()
plt.plot(x,y)
plt.xlabel('p*R')
plt.ylabel('ECDF')
plt.xscale('log')
# plt.xlim(0,1E20)
filename = os.path.join(plot_path, 'p_r_ECDF.png')
plt.savefig(filename, dpi=500, bbox_inches = 'tight')

plt.figure()
mask = np.isfinite(rat)
plt.plot(inttime[:220][mask], np.cumsum(0.5*rat[mask]))
(vxm3*data['volume'].sum(axis=0)).plot()
filename = os.path.join(plot_path, 'exp_volume_to_fit.png')
plt.savefig(filename, dpi=500, bbox_inches = 'tight')
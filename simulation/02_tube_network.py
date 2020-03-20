# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:40:26 2020

@author: firo
"""

import os
import xarray as xr
import numpy as np
import networkx as nx

drive = r"\\152.88.86.87\data118"

data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1200_dry_seg_aniso_sep'

sourceFolder  = os.path.join(data_path, processing_version)

pore_statistics = xr.load_dataset(os.path.join(sourceFolder, 'per_pore_statistics.nc'))
network_statistics = xr.load_dataset(os.path.join(sourceFolder, 'network_statistics.nc'))

#  Watts-Strogatz small world parameter
n = 300
p = 0.2
k = 6

r_ref = 5E-6 #m length scale of pore radii
h_ref = 5E-4 #m length scale of tube lengths
t_wait = 0 #0.5E-3 #waiting time before pore starts to get filled

R_inlet = 0 #resistance at inlet nodes (resistance of the yarn upstream)

eta = 1 #mPa*s dynamic viscosity of water
gamma = 72.6 #mN/m surface tension of water
theta = 50 #° contact angle
cos = np.cos(theta/180*np.pi)
px = 2.75E-6 #m

t_init = 1E-4 #s start time to stabilze simulation and avoid inertial regime, now irrelavent because flow rate is solved iteratively
tmax = 1E-2 #s
dt = 1E-4#s

# function to calculate the resistance of a full pore
def poiseuille_resistance(l, r, eta=eta):
    R = 8*eta*l/np.pi/r**4
    return R

# function to calculate the filling velocity considering the inlet resistance and tube radius
def capillary_rise(t, r, R0, cos = cos, gamma = gamma, eta =eta):
    dhdt = gamma*r*cos/2/eta/np.sqrt((R0*np.pi*r**4/8/eta)**2+gamma*r*cos*t/2/eta)
    return dhdt

# use capillary_rise2 because the former did not consider that R0 can change over time, should be irrelevant because pore contribution becomes quickly irrelevant , but still...
def capillary_rise2(r, R0, h, cos = cos, gamma = gamma, eta = eta):
    dhdt = 2*gamma*cos/(R0*np.pi*r**3+8*eta*h/r)
    return dhdt

def total_volume(h, r):
    V = (h*np.pi*r**2).sum()
    return V

def effective_resistance(R_nb):
    R = 1/(1/R_nb).sum()
    return R


def outlet_resistances(inlets, filled, R_full, net):
    
    R0 = np.zeros(len(filled))
    to_visit = np.zeros(len(filled), dtype=np.bool)
    to_visit[np.where(filled)] = True
    
    filled_inlets = inlets.copy()
    for nd in filled_inlets:
        if filled[nd] == False:
            filled_inlets.remove(nd)
    
    current_layer = filled_inlets
    to_visit[filled_inlets] = False
    
    while True in to_visit:
        next_layer = []
        for nd in current_layer:
            next_layer = next_layer + list(net.neighbors(nd))
        next_layer = list(np.unique(next_layer))
        for nd in next_layer:
            nnbb = list(net.neighbors(nd))
            R_nb = []
            for nb in nnbb:
                if nb in current_layer:
                    R_nb.append(R0[nb]+R_full[nb])
            R0[nd] = effective_resistance(np.array(R_nb))
            to_visit[nd] = False
        current_layer = next_layer.copy()
        for nd in current_layer:
            if filled[nd] == False:
                current_layer.remove(nd)
    return R0

def simulation(n=n, k=k, p=p, t_init=t_init, tmax=tmax, dt=dt, h_ref=h_ref, r_ref=r_ref, t_wait=t_wait, h_seq=False, r_seq=False, t_wait_seq=False, R_inlet = R_inlet):
    num_inlets = max(int(0.1*n),6)
    #net = nx.generators.random_graphs.watts_strogatz_graph(n, k, p)
    net = nx.generators.random_graphs.extended_barabasi_albert_graph(n, k, 0, p)
    # adj = nx.to_numpy_array(net).astype(np.bool)
    inlets = np.random.choice(np.arange(n), num_inlets)
    inlets = list(np.unique(inlets))
    filled = np.zeros(n, dtype = np.bool)
    active = inlets
    time = np.arange(t_init, tmax, dt)
    act_time = np.zeros(n)
   
    # initialization, currently equal distribution. Use experimental instead
    # initialize node properties
    h0 = h_ref * np.random.rand(n)+1E-5
    r = r_ref * np.random.rand(n)+1E-6
    t_w = t_wait * np.random.rand(n)
    
    if np.any(h_seq):
        h0 = h_seq[np.random.choice(h_seq.shape[0], n)]
    if np.any(r_seq):
        r = r_seq[np.random.choice(r_seq.shape[0], n)]
    if np.any(t_wait_seq):
        t_w = t_wait_seq[np.random.choice(t_wait_seq.shape[0], n)]
    
    R_full = poiseuille_resistance(h0, r)
    h = np.zeros(n)+1E-6
    R0 = np.zeros(n)
    R0[inlets] = R_inlet
    V=np.zeros(len(time))
    tt=0
    new_active = []
    for t in time:
        
        new_active = list(np.unique(new_active))
        if len(new_active)>0:
            for node in new_active:
                if filled[node] == True:
                    new_active.remove(node)
            act_time[new_active] = t + t_w[new_active]
            active = active + new_active
            R0 = outlet_resistances(inlets, filled, R_full, net)
        active = list(np.unique(active))
        new_active = []
        
        for node in active:
            if t>act_time[node]:
                h_old = h[node]
                #h[node] = h[node] + dt*capillary_rise(t-act_time[node], r[node], R0[node])
                h[node] = h_old + dt*capillary_rise2(r[node], R0[node], h_old)
                if h[node] >= h0[node]:
                    h[node] = h0[node]
                    filled[node] = True
                    active.remove(node)
                    new_active = new_active + list(net.neighbors(node))              
                
        V[tt] = total_volume(h, r)
        tt=tt+1
    return(time, V)

# get pore length distribution
h_seq = pore_statistics['pore_data'].sel(variables= 'arc_length').data
h_lim = pore_statistics['valuelimits'].sel(variables_limit = 'arc_length').data
h_seq = h_seq[h_seq<h_lim[1]]
h_seq = h_seq[h_seq>h_lim[0]]*px

# get equivalent radius distribution
r_seq = pore_statistics['pore_data'].sel(variables= 'median_area').data
r_lim = pore_statistics['valuelimits'].sel(variables_limit = 'median_area').data
r_seq = r_seq[r_seq<r_lim[1]]
r_seq = r_seq[r_seq>r_lim[0]]
r_seq = np.sqrt(r_seq/np.pi)*px

# get waiting time distribution
t_seq = network_statistics['deltatall'].data
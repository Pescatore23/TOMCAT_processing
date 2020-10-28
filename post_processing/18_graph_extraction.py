# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:51:25 2020

@author: firo
"""
import sys
LaurentCodePath=r"H:\10_Python\005_Scripts_from_others\Laurent\wicking_pnm"
if LaurentCodePath not in sys.path:
	sys.path.append(LaurentCodePath)
    
    
import os
from wickingpnm.model import PNM
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

njobs = 16
sourceFolder = r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep"

samples = []
for sample in os.listdir(sourceFolder):
    if sample[:3] == 'dyn':
        samples.append(sample[9:-3])

# if 'dyn_data_T3_025_4.nc' in samples: samples.remove('dyn_data_T3_025_4.nc')

print(samples)


def core_function(sample):
    pnm_params = {
           'data_path': r"A:\Robert_TOMCAT_3_netcdf4_archives\for_PNM",
          # 'data_path': r"A:\Robert_TOMCAT_3_netcdf4_archives\processed_1200_dry_seg_aniso_sep",
        'sample': sample,
    }
    
    pnm = PNM(**pnm_params)
    graph = pnm.graph
    
    return sample, graph


results = Parallel(n_jobs=njobs)(delayed(core_function)(sample) for sample in samples) 
# results = []
# for sample in samples:
#     print(sample)
#     results.append(core_function(sample))
    

np.save(r"R:\Scratch\305\_Robert\networks\networks_masked", results)

binsize = 0.045
centbins = np.arange(0,1,binsize)

cen_all = np.zeros((len(results), len(centbins)-1))
cen_025 = cen_all.copy()-1
cen_100 = cen_all.copy()-1
cen_300 = cen_all.copy()-1
cen_all_data = np.array([])
cen_025_data = cen_all_data.copy()
cen_100_data = cen_all_data.copy()
cen_300_data = cen_all_data.copy()

for i in range(len(results)):
    centr = nx.edge_betweenness_centrality(results[i][1])
    c = 0
    centralities = np.zeros(len(centr))
    for j in list(centr.keys()):
        centralities[c] = centr[j]
        c = c+1
    hist = np.histogram(centralities, centbins, density=True)
    cen_all[i,:] = hist[0]
    cen_all_data = np.concatenate([cen_all_data, centralities])
    
    if results[i][0][3:6] == '025':
        cen_025[i,:] = hist[0]
        cen_025_data = np.concatenate([cen_025_data, centralities])
    if results[i][0][3:6] == '100':
        cen_100[i,:] = hist[0]
        cen_100_data = np.concatenate([cen_100_data, centralities])
    if results[i][0][3:6] == '300':
        cen_300[i,:] = hist[0]
        cen_300_data = np.concatenate([cen_300_data, centralities])
        
# plt.hist(cen_all_data, centbins)
# plt.figure()
# plt.hist(cen_all_data, centbins)
# plt.yscale('log')

plt.figure()
plt.hist([cen_025_data, cen_100_data, cen_300_data], centbins, color= 'krb', density=True, label = ['0.25 mN/tex', '10 mN/tex', '30 mN/tex'])
plt.xlabel('norm. edge betweeness centrality')
plt.ylabel('density')
# plt.yscale('log')
plt.xlim(0,0.45)
plt.legend()

import tikzplotlib
path = r"C:\Zwischenlager\Paper\01_yarn_dynamics\Latex\XMT_data\edge_betweenes_centrality.tex"
tikzplotlib.save(path)

nodes_all = []
nodes_025 = []
nodes_100 = []
nodes_300 = []
for i in range(len(results)):
    nn = len(results[i][1].nodes)
    nodes_all.append(nn)
    if results[i][0][3:6] == '025':
        nodes_025.append(nn)
    if results[i][0][3:6] == '100':
        nodes_100.append(nn)
    if results[i][0][3:6] == '300':
        nodes_300.append(nn)
        
plt.figure()
plt.plot([2.5, 10, 30 ], [np.mean(nodes_025),np.mean(nodes_100), np.mean(nodes_300)], 'kx')
# plt.errorbar([2.5, 10, 30 ], [np.mean(nodes_025),np.mean(nodes_100), np.mean(nodes_300)], [np.std(nodes_025),np.std(nodes_100),np.std(nodes_300)], color='k', linestyle='')
plt.xlabel('yarn tension [mN/tex]')
plt.ylabel('nodes in network')
path = r"C:\Zwischenlager\Paper\01_yarn_dynamics\Latex\XMT_data\network_size.tex"
tikzplotlib.save(path)

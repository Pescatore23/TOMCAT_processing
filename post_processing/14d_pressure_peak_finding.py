# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:39:36 2020

@author: firo
"""

pc = dF/flux
pc = pc[np.isfinite(pc)]

pc = pc[pc<50000]
pc = pc[pc>-50000]

hist2 = np.histogram(pc, bins=1000)

pc_hist_x = hist2[1][:-1]
pc_hist_y = hist2[0]
mask5=pc_hist_x<-1000

pc_hist_x[mask5][np.argmax(pc_hist_y[mask5])]
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


y0=-0.35742
x0=-2030.32066
gamma=9642.88207
A=572690.71501
x0_dev = 47.51909

name = dyn_data.attrs['name']
x_hist = np.arange(-20000,20000, 1000)

plt.plot(-pc_hist_x/1000, pc_hist_y, 'k')
plt.plot(x_hist/1000, lorentz(-x_hist, gamma, x0, A, y0), 'r')
plt.xlim(-20,20)
plt.title(name+ '_x0_'+ str(x0)+'+-'+str(x0_dev)+'_gamma_'+str(gamma)+'_A_'+ str(A)+'_y0_'+str(y0))
plt.xlabel('capillary pressure [kPa]')
plt.ylabel('counts')
plt.ylim(-1, 70)

filename = r"C:\Zwischenlager\Paper\02_throat_physics\Latex\XMT_data\Resistance_fits\T3_025_3_III_p_fit_lorentz.tex"
tikzplotlib.save(filename)

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:33:06 2020

@author: firo
"""

k = 0.90076 #+-0.16429
m = 1.82679 #+-0.01027
xlog=np.arange(logflux.min(), logflux.max(), 0.1)
plt.plot(logflux, logdings, 'k.')
plt.plot(xlog, linear(xlog, k, m), 'r')

plt.xlabel('log10(flux [m3/s])')
plt.ylabel('log10(Fdot*Q)')



filename = r"C:\Zwischenlager\Paper\02_throat_physics\Latex\XMT_data\Resistance_fits\T3_025_3_III_p_fit.tex"
tikzplotlib.save(filename)
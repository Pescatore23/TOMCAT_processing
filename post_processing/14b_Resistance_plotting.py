# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:28:56 2020

@author: firo
"""

dest_folder = r"C:\Zwischenlager\Paper\02_throat_physics\Latex\XMT_data\Resistance_fits"

R=R[R>-5E18]
R=R[R<5E18]

hist = np.histogram(R, bins=500)

R_hist_y = hist[0]
R_hist_x = hist[1][:-1]

# get fit params from origin

# y0=
# x0=
# gamma=
# A=
x0_dev = 3.05916E16

name = dyn_data.attrs['name']
x_hist = np.arange(-2E18,2E18, 1E16)

plt.plot(R_hist_x, R_hist_y, 'k')
plt.plot(x_hist, lorentz(x_hist, gamma, x0, A, y0), 'r')
plt.xlim(-2E18,2E18)
plt.title(name+ '_x0_'+ str(x0)+'+-'+str(x0_dev)+'_gamma_'+str(gamma)+'_A_'+ str(A)+'_y0_'+str(y0))
plt.xlabel('resistance [Pa s m^-3]')
plt.ylabel('counts')
plt.ylim(-1, 15)

# filename = os.path.join(dest_folder, ''.join([name, '_R_fit.tex']))

# tikzplotlib.save(filename)

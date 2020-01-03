# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:08:33 2019

@author: firo
"""


from skimage.morphology import watershed


x_par = 'alpha [vx]'
y_par = 't0 [s]'
z_par = 'eccentricity_from_vertical_axis'


x_id = variables.index(x_par)
y_id = variables.index(y_par)
z_id = variables.index(z_par)
n = 3000

x = combined_data[:, x_id]
y = combined_data[:, y_id]
z = combined_data[:, z_id]

# filter data

#FIXME: add limits to np.where by np.logical_and


x=x[np.where(np.logical_and(x>=ylims[x_par][0], x<=ylims[x_par][1]))]
y=y[np.where(np.logical_and(x>=ylims[x_par][0], x<=ylims[x_par][1]))]
z=z[np.where(np.logical_and(x>=ylims[x_par][0], x<=ylims[x_par][1]))]

x=x[np.where(np.logical_and(y>=ylims[y_par][0], y<=ylims[y_par][1]))]
y=y[np.where(np.logical_and(y>=ylims[y_par][0], y<=ylims[y_par][1]))]
z=z[np.where(np.logical_and(y>=ylims[y_par][0], y<=ylims[y_par][1]))]

x=x[np.where(np.logical_and(z>=ylims[z_par][0], z<=ylims[z_par][1]))]
y=y[np.where(np.logical_and(z>=ylims[z_par][0], z<=ylims[z_par][1]))]
z=z[np.where(np.logical_and(z>=ylims[z_par][0], z<=ylims[z_par][1]))]

#plt.figure()
#plt.tricontourf(x,y,z, levels=50, cmap="inferno")
#heavily affected by outliers, try some smoothing or populating an image

#plt.figure()

im = np.zeros((n,n))

x_grid = np.arange(ylims[x_par][0], ylims[x_par][1], (ylims[x_par][1]-ylims[x_par][0])/n)
y_grid = np.arange(ylims[y_par][0], ylims[y_par][1], (ylims[y_par][1]-ylims[y_par][0])/n)

#z_x_sorted = z[np.argsort(x)]
#z_y_sorted = z[np.argsort(y)]
#
#x_sorted = np.sort(x)
#y_sorted = np.sort(y)

for kx in range(1,n):
    for ky in range(1,n):

        x_cand = np.where(np.logical_and(x>=x_grid[kx-1], x<x_grid[kx]))[0]
        y_cand = np.where(np.logical_and(y>=y_grid[ky-1], y<y_grid[ky]))[0]
        cand = np.intersect1d(x_cand, y_cand)
        
        if not len(cand)>0:
#            im[kx,ky] = im[kx, ky-1]
            continue
        
        z_vals = z[cand]
        z_val = np.mean(z_vals)
        if np.logical_and(z_val>=ylims[z_par][0], z_val<=ylims[z_par][1]):
            im[kx,ky] = z_val

im[np.isnan(im)] = 0

im_int = np.uint32(im/im.max()*2**16)

new_im = watershed(np.zeros(im_int.shape, dtype = np.uint8), im_int)

new_im = new_im/2**16*im.max()
#new_im[new_im<ylims[z_par][0]]=ylims[z_par][0]
#new_im[new_im>ylims[z_par][1]]=ylims[z_par][1]

plt.figure()
plt.contourf(x_grid, y_grid, new_im, min(n,100), cmap='magma')
plt.xlabel(x_par)
plt.ylabel(y_par)
plt.title(z_par)
plt.colorbar()

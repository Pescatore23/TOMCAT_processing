# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:22:17 2018

@author: firo
"""

import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from skimage import measure
from skimage.morphology import ball
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from joblib import Parallel, delayed
import multiprocessing as mp
import xarray as xr
from scipy import ndimage

num_cores = 12# mp.cpu_count()

drive = '//152.88.86.87/data118'
data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
processing_version = 'processed_1400_dry_seg_aniso_sep'


sourcefolder = os.path.join(drive, data_path, processing_version)
targetfolder = r'R:\Scratch\305\_Robert\Animations'



#ax = fig.add_subplot(111, projection='3d')
#
#dx=dy=dz=1
#ax.set_xlim(0, Shape[1]*dy)
#ax.set_ylim(0, Shape[0]*dx)
#ax.set_zlim(0, indexmax*dz)
#
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_zticks([])
#ax.set_aspect(indexmax/Shape[0])


def rendering(t, transitions, time, outfolder):
    
    
    if not os.path.exists(os.path.join(outfolder, ''.join(["time_",str(time[t]).zfill(4),'_s.png']))):
        
        Shape = transitions.shape
        
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    
        dx=dy=dz=1
        ax.set_xlim(0, Shape[1]*dy)
        ax.set_ylim(0, Shape[0]*dx)
        ax.set_zlim(0, Shape[2]*dz)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_aspect(np.int16(Shape[2]/Shape[0]))
    
        Stack = (transitions < t + 1 ).astype(np.uint8)*255
        Stack[transitions==0] =0
    
        if Stack.max() > 0:
            verts, faces, _, _ = measure.marching_cubes_lewiner(Stack,100,(1,1,1),step_size=1)
            mesh1 = Poly3DCollection(verts[faces],alpha=1, edgecolor='#00007B',linewidth = 0.001)
            mesh1.set_facecolor('#00008B')
            ax.add_collection3d(mesh1)
        ax.view_init(-10, 45)
        fig.savefig(''.join([outfolder,"/time_",str(time[t]).zfill(4),'_s.png']), format='png', dpi=600, transparent=True, facecolor='w', edgecolor='w',bbox_inches='tight')
        fig.clf()
        ax.cla()
        plt.close(fig)
 
    
def specific_rendering(sample, label, neighbours=False, z_limit = 1400, sourcefolder=sourcefolder, targetfolder=targetfolder, num_cores=num_cores):
    sample_file = ''.join(['dyn_data_',sample,'.nc'])
    sample_data = xr.load_dataset(os.path.join(sourcefolder, sample_file))
    label_matrix = sample_data['label_matrix'][:,:,:z_limit]
    pore = label_matrix == label
    time = sample_data['time'].data
    t_max=len(time)
    
    if not neighbours:
        bounding_box = ndimage.find_objects(pore)[0]
        outfolder = os.path.join(targetfolder, sample, ''.join(['label_',str(label)]))
    
    if neighbours:
        large_mask = np.zeros(label_matrix.shape, dtype=np.bool)
        mask = ndimage.morphology.binary_dilation(pore, structure = ball(2))
        labels = np.unique(label_matrix[mask])[1:]
        for color in labels:
            large_mask[label_matrix==color]=True
        bounding_box = ndimage.find_objects(large_mask)[0]
        outfolder = os.path.join(targetfolder, sample, ''.join(['label_',str(label),'_neighbours']))

    if not os.path.exists(os.path.join(targetfolder, sample))  :
        os.path.mkdir(os.path.join(targetfolder, sample))
    
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    
    transitions = sample_data['transition_matrix'][bounding_box].data
    
    Parallel(n_jobs=num_cores)(delayed(rendering)(t, transitions, time, outfolder) for t in range(t_max+1))
    
    
    
samples = os.listdir(sourcefolder)

samples = reversed(samples)

for sample in samples:
    if not sample[:3] == 'dyn': continue
    print(sample[9:-3])
    sample_data = xr.load_dataset(os.path.join(sourcefolder, sample))
    sample_name = sample_data.attrs['name']
    time = sample_data['time'].data
    
    outfolder = os.path.join(targetfolder, sample_name)
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
    transitions = sample_data['transition_matrix'].data
    t_max =  transitions.max()
    
    Parallel(n_jobs=num_cores)(delayed(rendering)(t, transitions[:,:,:1400], time, outfolder) for t in range(t_max+1))
#    for t in range(t_max+1)    :
#        rendering(t, transitions, time, outfolder)
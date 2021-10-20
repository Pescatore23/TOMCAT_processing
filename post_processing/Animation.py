# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:22:17 2018

@author: firo
"""

import matplotlib.pyplot as plt
# plt.ioff()
import numpy as np
from skimage import measure
from skimage.morphology import ball
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from joblib import Parallel, delayed
#import multiprocessing as mp
import xarray as xr
from scipy import ndimage
import robpylib

num_cores = 12# mp.cpu_count()
# temp_folder = r"Z:\users\firo\joblib_tmp"
# drive = '//152.88.86.87/data118'
temp_folder = None
drive = r"B:"
# data_path = os.path.join(drive, 'Robert_TOMCAT_3_netcdf4_archives')
# processing_version = 'processed_1200_dry_seg_aniso_sep'


# sourcefolder = os.path.join(drive, data_path, processing_version)
# sourcefolder = os.path.join(drive, "Robert_TOMCAT_5_netcdf4")
# sourcefolder = r"A:\Robert_TOMCAT_3_combined_archives\unmasked"
sourcefolder = "/home/firo/NAS/Robert_TOMCAT_4_netcdf4_split_v2_no_pore_size_lim"
# targetfolder = os.path.join(drive, "Robert_TOMCAT_5_netcdf4","Animations")
targetfolder = os.path.join(sourcefolder, "Animations")
# r'R:\Scratch\305\_Robert\Animations'



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


def rendering(t, transitions, time, outfolder, label = False, pore_object = False):
    
    if np.any(transitions==t):
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
            Stack[transitions==0] = 0
            
            if label:
                Stack2 = Stack*pore_object
                Stack[pore_object] = 0
        
            if Stack.max() > 0:
                verts, faces, _, _ = measure.marching_cubes_lewiner(Stack,100,(1,1,1),step_size=1)
                mesh1 = Poly3DCollection(verts[faces],alpha=1, edgecolor='#27408B',linewidth = 0.001)
                mesh1.set_facecolor('#3A5FCD')
                ax.add_collection3d(mesh1)
           
            if label:
                if Stack2.max() > 0:
                    verts, faces, _, _ = measure.marching_cubes_lewiner(Stack2,100,(1,1,1),step_size=1)
                    mesh2 = Poly3DCollection(verts[faces],alpha=1, edgecolor='#8B0000',linewidth = 0.001)
                    mesh2.set_facecolor('#CD0000')
                    ax.add_collection3d(mesh2)
                
                
            ax.view_init(-10, 45)
            fig.savefig(''.join([outfolder,"/time_",str(time[t]).zfill(4),'_s.png']), format='png', dpi=600, transparent=True, facecolor='w', edgecolor='w', bbox_inches='tight')
            fig.clf()
            ax.cla()
            plt.close(fig)
 
    
def specific_rendering(sample_data, label, neighbours=False, z_limit = 1400, sourcefolder=sourcefolder, targetfolder=targetfolder, num_cores=num_cores):
    # sample_file = ''.join(['dyn_data_',sample,'.nc'])
    # sample_data = xr.load_dataset(os.path.join(sourcefolder, sample_file))
    label_matrix = sample_data['label_matrix'][:,:,:z_limit].data
    
    pore = label_matrix == label
    
    if np.any(pore):
        time = sample_data['time'].data
        t_max=len(time)
    
    
        if not os.path.exists(os.path.join(targetfolder, sample)):
            os.mkdir(os.path.join(targetfolder, sample))
        
        if not neighbours:
            bounding_box = ndimage.find_objects(pore)[0]
            outfolder = os.path.join(targetfolder, sample, ''.join(['label_',str(label)]))
            transitions = sample_data['transition_matrix'][bounding_box].data * pore[bounding_box]
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
            Parallel(n_jobs=num_cores)(delayed(rendering)(t, transitions, time, outfolder) for t in range(t_max+1))
        
        if neighbours:
            large_mask = np.zeros(label_matrix.shape, dtype=np.bool)
            mask = ndimage.morphology.binary_dilation(pore, structure = ball(2))
            labels = np.unique(label_matrix*mask)[1:]
            for color in labels:
                large_mask[label_matrix==color]=True
    #        bounding_box = ndimage.find_objects(large_mask)[0]
            outfolder = os.path.join(targetfolder, sample, ''.join(['label_',str(label),'_neighbours']))
            transitions = sample_data['transition_matrix'].data * large_mask
            if not os.path.exists(outfolder):
                os.mkdir(outfolder)
            Parallel(n_jobs=num_cores)(delayed(rendering)(t, transitions, time, outfolder, label = label, pore_object = pore) for t in range(t_max+1))
        
        

    


    
    
    
    
# sample = 'T3_100_4_III'
# label = 5
##label =  17 #36 39 45 51
#LOI = [43]
##LOI = [27, 153, 11, 15, 20, 21, 22, 23, 36, 39, 48, 55, 56, 85, 174, 130, 149, 152, 158, 159, 162, 89, 92, 95, 106, 107]
#
##for label in LOI:
##    specific_rendering(sample, label, neighbours = True)
#    
#for label in LOI:
# specific_rendering(sample, label, z_limit=1200, neighbours = True)

samples = os.listdir(sourcefolder)

#
#samples = reversed(samples)
#
for sample in samples:
    if not sample[:3] == 'dyn': continue
# #     if not sample[9:-3] in robpylib.TOMCAT.INFO.samples_to_repeat: continue
# #     print(sample[9:-3])
    sample_data = xr.load_dataset(os.path.join(sourcefolder, sample))
    sample_name = sample_data.attrs['name']
    # if sample_name == 'T4_300_2_II': continue
    time = sample_data['time'].data
    LOI = sample_data['label'].data
    print(sample_name)
    # for label in LOI:
    #     specific_rendering(sample_data, label, neighbours=False, z_limit=1200)
    
    outfolder = os.path.join(targetfolder, sample_name, 'volume')
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
        
    transitions = sample_data['transition_matrix'].data
    t_max =  transitions.max()
    if t_max > 400: t_max=400
    
    interlaces, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(os.path.join("/home/firo/NAS/Robert_TOMCAT_4", sample_name, '06c_yarn_labels','interface_zone' ))
    transitions = transitions*interlaces
    interlaces = None
    
    
    Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(rendering)(t, transitions, time, outfolder) for t in range(t_max+1))
#    for t in range(t_max+1)    :
#        rendering(t, transitions, time, outfolder)
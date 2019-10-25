# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:36:15 2019

@author: firo
"""

import numpy as np
from scipy import ndimage
from skimage import measure
import pandas as pd
import os
from joblib import Parallel, delayed
import multiprocessing as mp
from skimage.morphology import ball

num_cores = mp.cpu_count()
# if you run in memory troubles, decrease num_cores. Should be unnecessary on a laptop

baseFolder = r'C:\Users\firo\Desktop\Neuer Ordner'
sourceFolder = os.path.join(baseFolder, 'dat_files')
targetFolder = os.path.join(baseFolder, 'result_tables')
OverWrite = False

if not os.path.exists(targetFolder):
    os.mkdir(targetFolder)


def reduced_pore_object(labels, label):
    pore = labels == label
    pore = np.uint8(pore)
    bounding_box = ndimage.find_objects(pore)[0]
    pore_object = pore[bounding_box].copy()
    return pore_object, bounding_box, label

def get_pore_props(pore_object, bounding_box, label):

#     scipy.ndimage and skimage.measure to analysis objects
    COM = ndimage.measurements.center_of_mass(pore_object)
    COM = COM + np.array([bounding_box[0].start, bounding_box[1].start, bounding_box[2].start])         
           
    inertia_tensor = measure._moments.inertia_tensor(pore_object)
    in_tens_eig = np.linalg.eigvalsh(inertia_tensor)
    volume = np.array(np.count_nonzero(pore_object))
    
    major_axis = 4*np.sqrt(in_tens_eig[-1])
    minor_axis = 4*np.sqrt(in_tens_eig[0])
        
    results = {'center_of_mass': COM,
               'major_axis': major_axis,
               'minor_axis': minor_axis,
               'volume': volume
            }  
    return label, results
    

samples = os.listdir(sourceFolder)


for sample in samples:
#    get sample name
    name = sample[:-4]
    targetname = ''.join([name,'.xlsx'])
    
# check if not already done
    if not OverWrite:
        if os.path.exists(os.path.join(baseFolder, targetFolder, targetname)):
            continue
        
# read dat-file and convert to numpy array    
    file = pd.read_csv(os.path.join(baseFolder, sourceFolder, sample))
    length = len(file[' VARIABLES= X'])
    sparse_matrix = np.zeros([length-1, 4], dtype = np.uint16)
    
    for i in range(1,length):
        line = file[' VARIABLES= X'][i]
        array = np.array([int(n) for n in line.split()])
        sparse_matrix[i-1, :] = array
        
    x_max = sparse_matrix[:,0].max()
    y_max = sparse_matrix[:,1].max()
    z_max = sparse_matrix[:,2].max()
    
    matrix = np.zeros([x_max, y_max, z_max], dtype = np.uint8)
    
    for i in range(sparse_matrix.shape[0]):
        x = sparse_matrix[i,0]-1     # first coefficient in python is 0, not 1 by convention
        y = sparse_matrix[i,1]-1
        z = sparse_matrix[i,2]-1
        matrix[x,y,z] = sparse_matrix[i,3]

# inflate beads and create mask
    spheres = matrix==1
    deposits = matrix==2
    spheres = measure.label(spheres, connectivity=1)
    
    balls = np.unique(spheres)
    balls = balls[1:]
    
    
    mask = np.zeros(spheres.shape, dtype = np.uint16)
    label_mask = np.zeros(spheres.shape, dtype = np.uint8)
    
    for sub_ball in balls:
#        this loop is a bit crude and slow, but works
        testmask = np.zeros(spheres.shape, dtype = np.uint8)
        
#        this is the bottleneck that takes the most time because scipy is not really optimized like numpy or scikit-image
        im = ndimage.morphology.binary_dilation(spheres == sub_ball, structure = ball(5.5))
                
        testmask[np.where(im)] = 1
        mask = mask + testmask
    
    label_mask[np.where(mask>1)]=1
    deposits = deposits*label_mask
    
# label all separate objects (contact area between beads)   
    label_matrix = measure.label(deposits, connectivity=1).astype(np.uint16)  
    labels, counts = np.unique(label_matrix, return_counts = True)
    labels = labels[1:]
    counts = counts[1:]

# filter out noisy stuff    
    threshold = 50    
    relevant_labels = labels[np.where(counts>threshold)]

# create subvolumes for more efficient calculation  
    pore_objects = []
    for label in relevant_labels:
        pore_objects.append(reduced_pore_object(label_matrix, label))

# do actual analysis of the contact areas in parallel   
    pore_props = Parallel(n_jobs=num_cores)(delayed(get_pore_props)(*pore_object) for pore_object in pore_objects)


# write results to excel file    
    labels = np.zeros(len(pore_props))
    major_axis = np.zeros(len(pore_props))
    minor_axis = np.zeros(len(pore_props))
    com_x = np.zeros(len(pore_props))
    com_y = np.zeros(len(pore_props))
    com_z = np.zeros(len(pore_props))
    volume = np.zeros(len(pore_props))
    
    for i in range(len(pore_props)):
        labels[i] = pore_props[i][0]
        com_x[i] = pore_props[i][1]['center_of_mass'][0]
        com_y[i] = pore_props[i][1]['center_of_mass'][1]
        com_z[i] = pore_props[i][1]['center_of_mass'][2]
        major_axis[i] = pore_props[i][1]['major_axis']
        minor_axis[i] = pore_props[i][1]['minor_axis']
        volume[i] = pore_props[i][1]['volume']
        
    data = pd.DataFrame({'label': labels,
                         'x_com': com_x,
                         'y_com': com_y,
                         'z_com': com_z,
                         'major_axis': major_axis,
                         'minor_axis': minor_axis,
                         'volume': volume})
    data.to_excel(os.path.join(baseFolder, targetFolder, targetname))
    
     
    
    
    
    
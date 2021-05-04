# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:04:59 2021

@author: firo
"""


import pandas as pd
import numpy as np
import robpylib
from scipy import ndimage
from skimage.morphology import cube
from joblib import delayed, Parallel
from collections import deque
import xarray as xr
import os
import traceback

drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_4')
destination = os.path.join(drive, 'Robert_TOMCAT_4_netcdf4')

# baseFolder = r"Z:\Robert_TOMCAT_4"
# destination = r"Z:\Robert_TOMCAT_4_netcdf4"
overWrite = False
temp_folder= r"Z:\users\firo\joblib_tmp"
# temp_folder = None

def extend_bounding_box(s, shape, pad=3):
            a = deque()
            for i, dim in zip(s, shape):
                start = 0
                stop = dim

                if i.start - pad >= 0:
                    start = i.start - pad
                if i.stop + pad < dim:
                    stop = i.stop + pad

                a.append(slice(start, stop, None))

            return tuple(a)

def make_skeleton(matrix, point_list, points):
    for point in point_list:
        y = points['X Coord'][point]
        x = points['Y Coord'][point]
        z = points['Z Coord'][point]
        matrix[x,y,z] = True
    return matrix

def dilate_skeleton(matrix):
    dt = ndimage.morphology.distance_transform_cdt(~matrix)
    matrix = dt<20
    return matrix

def function(shp, points, point_list):
    matrix = np.zeros(shp, dtype=np.bool)
    matrix = make_skeleton(matrix, point_list, points)
    matrix = dilate_skeleton(matrix)
    return matrix

def assign_pore(pore_object, fiber_object, label):
    mask = pore_object == label
    mask = ndimage.binary_dilation(input = mask, structure = cube(5))
    fiber_contacts = np.unique(fiber_object[mask])
    result = np.zeros(5, dtype=np.uint16)
    result[fiber_contacts+1] = 1
    result[0] =  label
    return result

def track_pore_affiliation(sample, relevant_pores, baseFolder=baseFolder, label_im = None):
    # path = r"A:\Robert_TOMCAT_4\T4_025_1_III\06_fiber_tracing\T4_025_1_III.CorrelationLines.xlsx"
# fiber_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\01a_weka_segmented_dry\classified"
# label_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\05b_labels"
    path = os.path.join(baseFolder, sample, '06_fiber_tracing', ''.join([sample,".CorrelationLines.xlsx"]))
    fiber_path = os.path.join(baseFolder, sample,'01a_weka_segmented_dry', 'classified')
    label_path = os.path.join(baseFolder, sample, '05b_labels_split')

    segments = pd.read_excel(path, sheet_name = 2)
    points = pd.read_excel(path, sheet_name = 1)
    nodes = pd.read_excel(path, sheet_name = 0)
    
    top_yarn = []
    bottom_yarn = []
    
    for segment in segments['Segment ID']:
        node_ids = [segments['Node ID #1'][segment], segments['Node ID #2'][segment]]
        position = nodes['Z Coord'][node_ids].mean()
        
        if position < 1012:
            top_yarn.append(segment)
        else:
            bottom_yarn.append(segment)
            
    top_points = ','.join(segments['Point IDs'][top_yarn])
    top_points = np.array([int(i) for i in top_points.split(',')])
    
    bottom_points = ','.join(segments['Point IDs'][bottom_yarn])
    bottom_points = np.array([int(i) for i in bottom_points.split(',')])
    
    fibers, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiber_path, track=False)
    fibers = fibers>0
    shp = fibers.shape
    
    yarns = Parallel(n_jobs=2, temp_folder = temp_folder)(delayed(function)(shp, points, point_list) for point_list in [top_points, bottom_points])
    
    labeled_fibers = np.bitwise_and(fibers, yarns[0]) + 2*np.bitwise_and(fibers, yarns[1])         
    
    if label_im is None:
        label_im, label_names =  robpylib.CommonFunctions.ImportExport.ReadStackNew(label_path, track=False)
    labels = np.unique(label_im)[1:]
    crude_pores = ndimage.find_objects(label_im)
    
    pores = deque()
    bounding_boxes = deque()
    
    for pore in crude_pores:
        if pore is not None: 
            if pore is relevant_pores:
           
                bb = extend_bounding_box(pore, shp, pad=4)
                pores.append(pore)
                bounding_boxes.append(bb)
        
    
    pore_assigned = Parallel(n_jobs=16, temp_folder = temp_folder)(delayed(assign_pore)\
        (label_im[bb], labeled_fibers[bb], label) for (bb, label) in zip(bounding_boxes, labels))
        
    pore_assigned = np.array(pore_assigned)
    
    # explanation pore_assigned: [label, 1, yarn1{0;1}, yarn2{0;1}, interlace{0,1}]
    # some (32) pores don't touch any fiber and 78 at interface -> revise
    
    # TO DO: label pore according to their affiliation and display in Avizo
    pore_affiliation = np.zeros(pore_assigned.shape[0], dtype=np.uint8)
    
    # populate vector: 1 -yarn1; 2-yarn2, 3-interlace
    # pore_affiliation[np.where(pore_assigned[:,2]>0)] = 1
    # pore_affiliation[np.where(pore_assigned[:,3]>0)] = 2
    # pore_affiliation[np.where(pore_assigned[:,4]>0)] = 3
    pore_affiliation[pore_assigned[:,2]>0] = 1
    pore_affiliation[pore_assigned[:,3]>0] = 2
    pore_affiliation[pore_assigned[:,4]>0] = 3   
    # top_pores = pore_assigned[pore_affiliation==1,0]
    # bottom_pores = pore_assigned[pore_affiliation==2,0]
    # interlace_pores = pore_assigned[pore_affiliation==3,0]
    # other_pores = pore_assigned[pore_affiliation==0,0]
    
    # new_label_im = np.zeros(label_im.shape, dtype=np.uint8)
    
    # for x in range(label_im.shape[0]):
    #     for y in range(label_im.shape[1]):
    #         for z in range(label_im.shape[2]):
    #             val = label_im[x,y,z]
    #             if val == 0: continue
    #             # check if top pore
    #             if val in other_pores:
    #                 new_label_im[x,y,z] = 4
    #             elif val in interlace_pores:
    #                 new_label_im[x,y,z] = 3              
    #             elif val in top_pores:
    #                 new_label_im[x,y,z] = 1
    #             elif val in bottom_pores:
    #                 new_label_im[x,y,z] = 2
    
    # path = os.path.join(baseFolder, sample, '07_pores_affiliated')
    # if not os.path.exists(path):
    #     os.mkdir(path)
        
    
    # robpylib.CommonFunctions.ImportExport.WriteStackNew(path, label_names, new_label_im)
        
    return pore_affiliation, labels

def sample_function(sample, baseFolder=baseFolder, destination=destination, overWrite=False):
    filename = ''.join(['pore_affiliation_', sample,'.nc'])
    path = os.path.join(destination, filename)
    
    if not os.path.exists(path) or overWrite:
        try:
            # metadata = xr.load_dataset(os.path.join(destination, ''.join(['pore_props_',sample,'.nc'])))
            metadata = xr.load_dataset(os.path.join(destination, ''.join(['dyn_data_',sample,'.nc'])))
            relevant_pores = metadata['label'].data
            pore_affiliation, labels = track_pore_affiliation(sample, relevant_pores, baseFolder, label_im = metadata['label_matrix'].data)
            
            data = xr.Dataset({'pore_affiliation': ('label', pore_affiliation)},
                              coords= {'label': labels})
                              # attrs={'explanation': '1 - top yarn, 2 - bottom yarn, 3 - interlace, 0 - not in contact'})
            data.attrs = metadata.attrs
            data.attrs['explanation'] = '1 - top yarn, 2 - bottom yarn, 3 - interlace, 0 - not in contact'
            
            filename = ''.join(['pore_affiliation_', sample,'.nc'])
            path = os.path.join(destination, filename)
            
            data.to_netcdf(path)
            return 'completed'
        except Exception:# as e:
            # return
            traceback.print_exc()
            
    else:
        return 'already done'
        
    
samples = os.listdir(baseFolder)

if '.DS_Store' in samples:
    samples.remove('.DS_Store')

num_jobs = 8
results = Parallel(n_jobs=num_jobs, temp_folder=temp_folder)(delayed(sample_function)(sample, overWrite=overWrite) for sample in samples)

for state in zip(samples, results):
    print(state)
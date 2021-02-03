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

path = r"A:\Robert_TOMCAT_4\T4_025_1_III\06_fiber_tracing\T4_025_1_III.CorrelationLines.xlsx"
fiber_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\01a_weka_segmented_dry\classified"
label_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\05b_labels"

temp_folder= r"Z:\users\firo\joblib_tmp"

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

fibers, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiber_path)
fibers = fibers>0
shp = fibers.shape


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

yarns = Parallel(n_jobs=2, temp_folder = temp_folder)(delayed(function)(shp, points, point_list) for point_list in [top_points, bottom_points])

labeled_fibers = np.bitwise_and(fibers, yarns[0]) + 2*np.bitwise_and(fibers, yarns[1])

# TO DO: assign pores to top or bottom

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
        

label_im, label_names =  robpylib.CommonFunctions.ImportExport.ReadStackNew(label_path)
labels = np.unique(label_im)[1:]
crude_pores = ndimage.find_objects(label_im)

pores = deque()
bounding_boxes = deque()

for pore in crude_pores:
    if pore is not None: 
        bb = extend_bounding_box(pore, shp, pad=4)
        pores.append(pore)
        bounding_boxes.append(bb)
        
def assign_pore(pore_object, fiber_object, label):
    mask = pore_object == label
    mask = ndimage.binary_dilation(input = mask, structure = cube(5))
    fiber_contacts = np.unique(fiber_object[mask])
    result = np.zeros(5, dtype=np.uint16)
    result[fiber_contacts+1] = 1
    result[0] =  label
    return result

pore_assigned = Parallel(n_jobs=32, temp_folder = temp_folder)(delayed(assign_pore)\
    (label_im[bb], labeled_fibers[bb], label) for (bb, label) in zip(bounding_boxes, labels))
    
pore_assigned = np.array(pore_assigned)

# explanation pore_assigned: [label, 1, yarn1{0;1}, yarn2{0;1}, interlace{0,1}]
# some (32) pores don't touch any fiber and 78 at interface -> revise

# TO DO: label pore according to their affiliation and display in Avizo

# new_label_im = np.zeros(label_im.shape, dtype=np.uint8)

# for label in pore_assigned[:,0]:
#     color = labels[label]
#     if pore_assigned[label,2] > 0:
#         new_label_im[np.where(label_im==label)] = 2
#     if pore_assigned[label,3] > 0:
#         new_label_im[np.where(label_im==label)] = 3
#     if pore_assigned[label,4] > 0:
#         new_label_im[np.where(label_im==label)] = 4  
#     else:
#         new_label_im[np.where(label_im==label)] = 1 
        

# robpylib.CommonFunctions.ImportExport.WriteStackNew(r"R:\Scratch\305\_Robert\interlace_label_test", label_names, new_label_im)





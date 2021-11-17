# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 16:09:41 2021

@author: firo
"""

import os
from scipy import ndimage
import numpy as np
import pandas as pd
import robpylib
from joblib import Parallel, delayed
# import xarray as xr
from skimage.morphology import square, disk


drive = '//152.88.86.87/data118'
baseFolder = os.path.join(drive, 'Robert_TOMCAT_4')
destination = os.path.join(drive, 'Robert_TOMCAT_4_netcdf4_split_v2')

baseFolder = r"A:\Robert_TOMCAT_4"
destination = r"A:\Robert_TOMCAT_4_netcdf4_split_v2"
overWrite = False
temp_folder= r"Z:\users\firo\joblib_tmp"


def make_skeleton(matrix, point_list, points):
    for point in point_list:
        y = points['X Coord'][point]
        x = points['Y Coord'][point]
        z = points['Z Coord'][point]
        matrix[x,y,z] = True
    return matrix

def dilate_skeleton(matrix):
    dt = ndimage.morphology.distance_transform_edt(~matrix)
    matrix = dt<12
    return matrix

def trace_function(shp, points, segments, fiber):
    matrix = np.zeros(shp, dtype=bool)
    point_ids = np.array([int(i) for i in segments['Point IDs'][fiber].split(',')])
    iold=point_ids[0]
    for i in point_ids[1:]:
        X1 = np.array([points['X Coord'][iold], points['Y Coord'][iold], points['Z Coord'][iold]])
        X2 = np.array([points['X Coord'][i], points['Y Coord'][i], points['Z Coord'][i]])
        T = X2-X1
        DX = int(np.sqrt((T**2).sum()))
        for t in range(DX):
            Xz = t*T/DX + X1
            Y = np.int16(Xz[0])
            X = np.int16(Xz[1])
            Z = np.int16(Xz[2])
            matrix[X,Y,Z] = True
        
        # T = X2-X1            
        # DZ = T[2]
        # for z in range(DZ):
        #     Xz = z*T/DZ + X1
        #     Y = np.int16(Xz[0])
        #     X = np.int16(Xz[1])
        #     Z = np.int16(Xz[2])
        #     matrix[X,Y,Z] = True
        iold = i
    return matrix

def make_traces_parallel(shp, yarn, segments, points):
    matrix = np.zeros(shp, dtype = np.uint8)
    colors= 1 + np.arange(len(yarn))
    results = Parallel(n_jobs = 16, temp_folder = temp_folder)(delayed(trace_function)(shp, points, segments, fiber) for fiber in yarn)
    for (result, color) in zip(results, colors):
        matrix[result.astype(bool)] = color
    return matrix, colors
       

def make_traces(matrix, yarn, segments, points):
    count = 0
    for fiber in yarn:
        count = count+1
        point_ids = np.array([int(i) for i in segments['Point IDs'][fiber].split(',')])
        iold=point_ids[0]
        for i in point_ids[1:]:
            X1 = np.array([points['X Coord'][iold], points['Y Coord'][iold], points['Z Coord'][iold]])
            X2 = np.array([points['X Coord'][i], points['Y Coord'][i], points['Z Coord'][i]])
            T = X2-X1            
            DZ = T[2]
            for z in range(DZ):
                Xz = z*T/DZ + X1
                matrix[Xz] = count
            iold = i
        
    return matrix, count 

def expand_trace(matrix, radius, color):
    dt = ndimage.morphology.distance_transform_edt(~(matrix==color))
    return dt<radius

def make_fibers(shp, yarn, segments, points, radius=10):
    # matrix, count = make_traces(matrix, yarn, segments, points)
    print('make fiber traces')
    matrix, fiber_colors = make_traces_parallel(shp, yarn, segments, points)
    print('expand traces to fibers')
    results = Parallel(n_jobs=5, temp_folder = temp_folder)(delayed(expand_trace)(matrix, radius, color) for color in fiber_colors)
    matrix = np.zeros(shp, dtype=np.uint8)
    for (result, color) in zip(results, fiber_colors):
        matrix[result.astype(bool)] = color
    return matrix

def function(shp, points, point_list, segments):
    matrix = np.zeros(shp, dtype = bool)
    # matrix = make_skeleton(matrix, point_list, points)
    # matrix = dilate_skeleton(matrix)
    matrix = make_fibers(matrix, point_list, points, segments)
    return matrix

def yarn_labeling(im):
    close = ndimage.morphology.binary_closing(im, iterations = 10)
    dilate = ndimage.morphology.binary_dilation(close, structure = disk(5))
    return dilate

def track_yarn_affiliation(sample, baseFolder=baseFolder):
    # path = r"A:\Robert_TOMCAT_4\T4_025_1_III\06_fiber_tracing\T4_025_1_III.CorrelationLines.xlsx"
# fiber_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\01a_weka_segmented_dry\classified"
# label_path = r"A:\Robert_TOMCAT_4\T4_025_1_III\05b_labels"
    path = os.path.join(baseFolder, sample, '06_fiber_tracing', ''.join([sample,".CorrelationLines.xlsx"]))
    fiber_path = os.path.join(baseFolder, sample,'01a_weka_segmented_dry', 'classified')
    targetFolder = os.path.join(baseFolder, sample, '06c_yarn_labels')
    if not os.path.exists(targetFolder):
        os.mkdir(targetFolder)
    segments = pd.read_excel(path, sheet_name = 2)
    points = pd.read_excel(path, sheet_name = 1)
    # nodes = pd.read_excel(path, sheet_name = 0)
    
    # top_yarn = []
    # bottom_yarn = []
    
    # for segment in segments['Segment ID']:
    #     node_ids = [segments['Node ID #1'][segment], segments['Node ID #2'][segment]]
    #     position = nodes['Z Coord'][node_ids].mean()
        
    #     if position < 1012:
    #         top_yarn.append(segment)
    #     else:
    #         bottom_yarn.append(segment)
            
    # top_points = ','.join(segments['Point IDs'][top_yarn])
    # top_points = np.array([int(i) for i in top_points.split(',')])
    
    # bottom_points = ','.join(segments['Point IDs'][bottom_yarn])
    # bottom_points = np.array([int(i) for i in bottom_points.split(',')])
    
    fibers, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(fiber_path, track=False)
    fibers = fibers>0
    shp = fibers.shape
    fibers = None
    # ncpath = os.path.join(ncfolder)
    
    # yarns = Parallel(n_jobs=2, temp_folder = temp_folder)(delayed(function)(shp, points, point_list) for point_list in [top_points, bottom_points])
    # yarn1 = yarns[0].astype(np.uint8)
    # yarn2 = yarns[1].astype(np.uint8)
    
    both_yarn = list(segments['Segment ID'])
    yarns = make_fibers(shp, both_yarn, segments, points)
    # yarn1 = make_fibers(shp, top_yarn, segments, points)
    # yarn2 = make_fibers(shp, bottom_yarn, segments, points)
    
    targetfiber = os.path.join(targetFolder, 'all_fibers')
    # targetfiber1 = os.path.join(targetFolder,'yarn1_fibers')
    # targetfiber2 = os.path.join(targetFolder,'yarn2_fibers')
    
    robpylib.CommonFunctions.ImportExport.WriteStackNew(targetfiber, names, yarns)
    
    # robpylib.CommonFunctions.ImportExport.WriteStackNew(targetfiber1, names, yarn1)
    # robpylib.CommonFunctions.ImportExport.WriteStackNew(targetfiber2, names, yarn2)
    
    # yarn1 = yarn1>0
    # yarn2 = yarn2>0
    
    # results = Parallel(n_jobs = 16, temp_folder = temp_folder)(delayed(yarn_labeling)(yarn1[:,:,z]) for z in range(yarn1.shape[2]))
    # label = np.array(results).transpose(1,2,0).astype(np.uint8)
    # target1 = os.path.join(targetFolder,'yarn1_small')
    # if not os.path.exists(target1):
    #     os.mkdir(target1)
    # robpylib.CommonFunctions.ImportExport.WriteStackNew(target1, names, label)
    
    # results = Parallel(n_jobs = 16, temp_folder = temp_folder)(delayed(yarn_labeling)(yarn2[:,:,z]) for z in range(yarn1.shape[2]))
    # label = np.array(results).transpose(1,2,0).astype(np.uint8)
    # target2 = os.path.join(targetFolder,'yarn2_small')
    # if not os.path.exists(target2):
    #     os.mkdir(target2)
    # robpylib.CommonFunctions.ImportExport.WriteStackNew(target2, names, label)

samples = os.listdir(baseFolder)

if '.DS_Store' in samples:
    samples.remove('.DS_Store')
# track_yarn_affiliation('T4_025_1_III')
# num_jobs = 2
# results = Parallel(n_jobs=num_jobs, temp_folder=temp_folder)(delayed(track_yarn_affiliation)(sample) for sample in samples)    

for sample in samples:
    print(sample)
    if os.path.exists(os.path.join(baseFolder, sample, '06c_yarn_labels', 'all_fibers' )): 
        print('already done')
        continue

    track_yarn_affiliation(sample)
    
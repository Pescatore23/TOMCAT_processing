# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:00:14 2021

@author: firo
"""

import os
from collections import deque
import networkx as nx
import robpylib
import imageio
import numpy as np
import scipy as sp
from skimage import morphology as skmorph
from skimage.morphology import cube
from joblib import Parallel, delayed
import multiprocessing as mp
import xarray as xr

def yarn_pores(fiberFolder, targetFolder, name):
    fiber = imageio.imread(os.path.join(fiberFolder,name))
    fiber = fiber>0

    hull = skmorph.convex_hull_image(fiber)
    pores = np.bitwise_xor(hull, fiber)
    pores = skmorph.remove_small_objects(pores, min_size=5, connectivity=1)
    pores = (pores*255).astype(np.uint8)
    imageio.imsave(os.path.join(targetFolder,name), pores)

def label_function(struct, pore_object, label, labels):
    mask = pore_object == label
    connections = deque()

    mask = sp.ndimage.binary_dilation(input = mask, structure = struct(3))
    neighbors = np.unique(pore_object[mask])[1:]

    for nb in neighbors:
        if nb != label:
            if nb in labels:
                conn = (label, nb)   
                connections.append(conn)

    return connections

def extract_throat_list(label_matrix, labels): 
    """
    inspired by Jeff Gostick's GETNET

    extracts a list of directed throats connecting pores i->j including a few throat parameters
    undirected network i-j needs to be calculated in a second step
    """

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

    im = label_matrix

    struct = cube # ball does not work as you would think (anisotropic expansion)

    crude_pores = sp.ndimage.find_objects(im)

    # throw out None-entries (counterintuitive behavior of find_objects)
    pores = deque()
    bounding_boxes = deque()
    for pore in crude_pores:
        if pore is not None: bb = extend_bounding_box(pore, im.shape)
        if pore is not None and len(np.unique(im[bb])) > 2:
            pores.append(pore)
            bounding_boxes.append(bb)

    connections_raw = Parallel(n_jobs = 32)(
        delayed(label_function)\
            (struct, im[bounding_box], label, labels) \
            for (bounding_box, label) in zip(bounding_boxes, labels)
    )

    # clear out empty objects
    connections = deque()
    for connection in connections_raw:
        if len(connection) > 0:
            connections.append(connection)

    return np.concatenate(connections, axis = 0)

def generate_graph(label_matrix, labels):
    throats = extract_throat_list(label_matrix, labels)
    graph = nx.Graph()
    graph.add_edges_from(np.uint16(throats[:,:2]))
    Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    graph = graph.subgraph(Gcc[0])
    return graph

sourcefolder1 = "A:\Robert_TOMCAT_1"
sourcefolder2 = "A:\Robert_TOMCAT_2"
sourcefolder3 = "A:\Robert_TOMCAT_3"

sourcefolders = [sourcefolder1, sourcefolder2, sourcefolder3]

sample_paths = []
samples = []
for folder in sourcefolders:
    samples_in_folder = os.listdir(folder)
    for sample in samples_in_folder:
        samples.append(sample)
        sample_paths.append(os.path.join(folder,sample))


for i in range(len(samples)):
    sample = samples[i]
    if sample[0] == '1': continue
    sample_path = sample_paths[i]
    fiber_path = os.path.join(sample_path, '01a_weka_segmented_dry', 'classified')
    void_path = os.path.join(sample_path, '04a_void_space')
    
    # first run: create void image to be separated in ImageJ
    if not os.path.exists(void_path):
        print(sample)
        print('first run')
        print(void_path)
        os.mkdir(void_path)
        fibernames = os.listdir(fiber_path)
        if 'Thumbs.db' in fibernames: fibernames.remove('Thumbs.db')
    
        num_cores=mp.cpu_count()
        Parallel(n_jobs=32)(delayed(yarn_pores)(fiber_path, void_path, name) for name in fibernames)
    
    # second run: extract network and save to nc
    label_path = os.path.join(sample_path, '05_labels')
    if os.path.exists(label_path):
        network_file = os.path.join(sample_path, ''.join([sample,'_network.nc']))
        if not os.path.exists(network_file):
            print(sample)
            print('second run')
            print(label_path)
            Stack, names = robpylib.CommonFunctions.ImportExport.ReadStackNew(label_path)
            label_matrix = Stack
            labels = np.unique(label_matrix)[1:]
            graph = generate_graph(label_matrix, labels)
            adj_matrix = nx.to_numpy_array(graph)
            nodes = list(graph.nodes)
            
            if sample[0] == 'T':
                twist = 200
                filament_count = 32
                tension = np.uint16(sample[3:6])
                sample_id = np.uint8(sample[7])
                last_3 = sample[-3:]
                if last_3 == 'III':
                    repeat = 3
                elif last_3 == '_II':
                    repeat = 2
                else:
                    repeat = 1
            else:
                filament_count = np.uint16(sample[:2])
                twist = np.uint16(sample[3:6])
                tension = np.uint16(sample[7:10])
                repeat = np.uint16(sample[-1])
                sample_id = sample[-2]
                
            data = xr.Dataset({'label_matrix': (['x','y','z'], label_matrix),
                               'adj_matrix': (['node','node'], adj_matrix)},
                              coords = {'node': nodes,
                                        'x': np.arange(label_matrix.shape[0]),
                                        'y': np.arange(label_matrix.shape[1]),
                                        'z': np.arange(label_matrix.shape[2])},
                             attrs = {'twist': twist,
                                      'tension': tension,
                                      'filament_count': filament_count,
                                      'ID': sample_id,
                                      'repeat': repeat,
                                      'pixel_size': 2.75E-6,
                                      'name': sample}) 
            data.to_netcdf(network_file)                             
                              
                
            
        
    # TO DO: analysis of network for all examined samples
    # try to find characteristic network properties depending on twisitng state
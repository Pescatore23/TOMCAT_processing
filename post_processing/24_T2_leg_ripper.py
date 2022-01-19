# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:51:58 2022

@author: firo
"""

# import numpy as np
from robpylib.CommonFunctions.ImportExport import WriteStackNew
from robpylib.CommonFunctions.ImportExport import ReadStackNew
import os
from joblib import delayed, Parallel

# leg coordiantes (x,y,z, width, length, height)

leg_coords = {
              'R_m4_33_050_2': ((334, 120, 0, 256, 330, 869), 
                                (438, 590, 0, 208, 268, 869),
                                (130, 512, 1098, 290, 372, 921),
                                (564, 214, 1098, 358, 296, 921))
               }



baseFolder = r"N:\Dep305\Robert_Fischer\Robert_TOMCAT_2"
samples = ['R_m4_33_050_2']
jobs = 8

def rip_legs(sample, i, Stack, names, sourceFolder, timestepname):
    
    x,y,z,dx,dy,dz = leg_coords[sample][i]
    leg = Stack[y:y+dy,x:x+dx,z:z+dz]
    legnames = names[z:z+dz]
    
    
    legfolder = ''.join([timestepname,'_leg_',str(i)])
    targetfolder = os.path.join(sourceFolder, legfolder)
    
    WriteStackNew(os.path.join(targetfolder), legnames, leg, track=False)
    
    
def timestep_operation(sample, sourceFolder, timestepname):
    
    Stack, names = ReadStackNew(os.path.join(sourceFolder, timestepname), track=False)
    for i in range(4):
        rip_legs(sample, i, Stack, names, sourceFolder, timestepname)
        
    return True
        
def sample_operation(sample, baseFolder=baseFolder):
    sourceFolder = os.path.join(baseFolder, sample, '02_pystack_registered')
    timestepnames = os.listdir(sourceFolder)
    
    Parallel(n_jobs = jobs, temp_folder = r"Z:\users\firo\joblib_tmp")(delayed(timestep_operation)(sample, sourceFolder, timestepname) for timestepname in timestepnames)
              
for sample in samples:
    sample_operation(sample)
    
    
    
    
    
    
    
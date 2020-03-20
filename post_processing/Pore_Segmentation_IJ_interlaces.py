"""
Created on Fri Aug 24 09:17:06 2018

To be run inside IJ


@author: firo
"""

import sys
homeCodePath="H:\\10_Python\\008_TOMCAT_processing\\tomcat-processing\\processing_final_version"
if homeCodePath not in sys.path:
	sys.path.append(homeCodePath)
from ij import IJ
import RobertFijiFunctions as rff 
import os
import time

network_location = '//152.88.86.87/data118'

#baseFolder=r'X:\TOMCAT3_processing_1'
#baseFolder = r'Y:\TOMCAT_3'
#baseFolder = r'W:\Robert_TOMCAT_3_Part_2'
#baseFolder = r'F:\Zwischenlager_Robert\TOMCAT_3'
baseFolder = os.path.join(network_location, 'Robert_TOMCAT_4')

procFolder="04b_split_void"
#procFolder = '04a_void_space_from_5'
#procFolder="04a_void_space_test"
#procFolder = '04a_void_space_temp_mean'

outname = 'pores_'
samples=os.listdir(baseFolder)

settings="disconnection=0.7 xsize=1.5 ysize=1.5 zsize=1 sigma=[] holes=0 particles=0 entire evaluation=[3D volumetric processing]"
#settings="disconnection=0.7 xsize=1 ysize=1 zsize=1 sigma=[] holes=0 particles=0 entire evaluation=[3D volumetric processing]"


def labelImage(procfolder,targetFolder,outname,settings):
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)  
    imp, names = rff.openSilentStack(procfolder)
    label=IJ.run(imp, "Disconnect Particles", settings)
    savecmd=''.join(["format=TIFF name=",outname," save=[",targetFolder,'\\',outname,"0000.tif]"])
    IJ.run(label, "Image Sequence... ", savecmd)
    IJ.run("Close All")
       
    
time0=time.time()   
length=len(samples)
step=0

targetName = "05a_separated"
#targetName = '05a_separated_from_5'
#targetName = '05a_separated_temp_mean'

#if baseFolder[-1] == '2': targetName = "04a_separated"
#samples = ['T3_300_3', 'T3_025_1', 'T3_025_4']
#samples = ['T3_025_9_III']
#samples = reversed(samples)
parts = ['part_1', 'part_2', 'part_3']
for sample in samples:
    step=step+1
    if not sample[1]=='4': continue
    if sample == 'T4_025_1': continue
    print(sample,' (',step,'/',length,')')
    if not os.path.exists(os.path.join(baseFolder, sample, procFolder)): 
    	print('no void space data available')
    	continue
    
    sourceFolder=os.path.join(baseFolder,sample,procFolder)
    targetFolder=os.path.join(baseFolder,sample, targetName, settings)
    for part in parts:
    	skipFlag=False
    	sampleFolder = os.path.join(sourceFolder, part)
    	subFolder = os.path.join(targetFolder, part)
    	if os.path.exists(subFolder):
	        f1=os.listdir(subFolder)
	        if len(f1)>500:
	        	print(sample, ' skipped (already calculated)')
	        	skipFlag=True
    	print('reached that point')
    	if not skipFlag:
	   		print('reached this point')
	   		labelImage(sampleFolder,subFolder,outname,settings)

print(time.time()-time0)
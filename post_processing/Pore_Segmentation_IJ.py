"""
Created on Fri Aug 24 09:17:06 2018

To be run inside IJ


@author: firo
"""

import sys
homeCodePath="H:\\10_Python\\008_TOMCAT_processing"
if homeCodePath not in sys.path:
	sys.path.append(homeCodePath)
from ij import IJ
import RobertFijiFunctions as rff 
import os
import time

network_location = '//152.88.86.87/data118'

baseFolder=r'U:\TOMCAT_3_segmentation'
baseFolder = r'Z:\Robert_TOMCAT_3'
baseFolder = os.path.join(network_location, 'Robert_TOMCAT_3_Part_2')

#procFolder="04a_void_space"
#procFolder="04a_void_space_test"
procFolder = '04a_void_space_temp_mean'


outname = 'pores_'
samples=os.listdir(baseFolder)

#settings="disconnection=0.7 xsize=1.5 ysize=1.5 zsize=1 sigma=[] holes=0 particles=0 entire evaluation=[3D volumetric processing]"
settings="disconnection=0.7 xsize=1 ysize=1 zsize=1 sigma=[] holes=0 particles=0 entire evaluation=[3D volumetric processing]"


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
targetName = '05a_separated_temp_mean'

#if baseFolder[-1] == '2': targetName = "04a_separated"

#samples = reversed(samples)
for sample in samples:
    skipFlag=False
    step=step+1
    if sample[1]=='4': continue
    if not os.path.exists(os.path.join(baseFolder, sample, procFolder)): continue
    print(sample,' (',step,'/',length,')')
    sampleFolder=os.path.join(baseFolder,sample,procFolder)
    targetFolder=os.path.join(baseFolder,sample, targetName, settings)
    if os.path.exists(targetFolder):
        f1=os.listdir(targetFolder)
        if len(f1)>2000:
            print(sample, ' skipped (already calculated)')
            skipFlag=True
    if not skipFlag: labelImage(sampleFolder,targetFolder,outname,settings)

print(time.time()-time0)
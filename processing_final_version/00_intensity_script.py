# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:05:43 2018

Corrects image intensity profile over the sample height introduced by the phase filter
Takes the difference of the intensity maxima (= on-set of fiber phase) (test for robustness) and shifts the intensity


current state: 





@author: firo
"""
import os
import robpylib
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp

num_cores=mp.cpu_count()
num_cores = 32
temp_folder = r"Z:\users\firo\joblib_tmp"
#baseFolder="T:\\DATA\\2018_07_12_Test_Image_Processing_Tomcat"
#baseFolder="T:\\Samples_with_Water_inside"
#baseFolder="R:\\Scratch\\305\\_Robert\\TOMCAT3_yarn"
#baseFolder=r"T:\Zwischenlager\disk1"
#baseFolder=r"O:\disk2"
baseFolder = r"Z:\Robert_TOMCAT_3"
# baseFolder = r"F:\Zwischenlager_Robert\TOMCAT_3"
baseFolder = "E:\Robert_TOMCAT_3b"

repeats = robpylib.TOMCAT.INFO.samples_to_repeat
#newDiskfolder=r'F:\Zwischenlager_Robert\TOMCAT_3'
# newDiskfolder=r'X:\TOMCAT3_processing_1'
# newBaseFolder=r'Y:\TOMCAT_3'
newDiskfolder = "E:\Robert_TOMCAT_3b"

excluded_samples=[
 'T3_100_4',        #FOV moved during acquisition
 'T3_300_6',        # water only in little pores
 'T3_300_3_III',    #no water
 'T4_025_3_III',    #wet from start
 'T3_300_8',        #FOV moved
 'T3_025_7',        #water only in small pores
 'T3_025_4_III',    #little water, data set incomplete, missing time steps reconstructed at PSI-ra-cluster, but only little water -> not reasonable to process?!
 'T3_025_3_II'      #little, water wet with acetone
 ]

parallel=True

def makesamplelist(baseFolder, **kwargs):
#    This function is a stub, expand it to apply sophisticated sample list creation on all processing steps
    samples=os.listdir(baseFolder)
#    samples=["32_200_025H2_cont"]
    return samples


def test_recalculate(sourceFolder,targetFolder,z,OverWrite=False):
    breakFlag=False
    first_scan=os.listdir(sourceFolder)[0]
    if os.path.exists(targetFolder):   
        if len(os.listdir(targetFolder))>0:
            last_scan=os.listdir(targetFolder)[-1]
            lst_nr=last_scan[-5:]
            test_name=os.listdir(os.path.join(sourceFolder,first_scan))[z]
            test_name=list(test_name)
            test_name[-24:-19]=lst_nr
            test_name=''.join(test_name)
            if not OverWrite:
                if test_name in os.listdir(os.path.join(targetFolder,last_scan)):
                    breakFlag=True
    return breakFlag

def corefunction(z,sourceFolder,targetFolder_med):
    breakFlag=test_recalculate(sourceFolder,targetFolder_med,z)
    if not breakFlag:
        Tstack, names, scans = robpylib.CommonFunctions.ImportExport.OpenTimeStack(sourceFolder, z)
        Tstack=np.int32(Tstack)
        Tstackmed=Tstack
        refmed=np.median(Tstack[:,:,0])
        for j in range(Tstack.shape[2]):
            Tstackmed[:,:,j]=Tstackmed[:,:,j]-(np.median(Tstackmed[:,:,j])-refmed)
        Tstackmed[Tstackmed<0]=0
        Tstackmed=np.uint16(Tstackmed)            
        robpylib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder_med,Tstackmed,names, scans)


def correctIntensityProfile(sample, newDiskfolder=newDiskfolder, parallel=parallel):
    sourceFolder = os.path.join(baseFolder, sample, "00_raw")  #do this before registration in the future (replace by "00_raw")

    targetFolder_med = os.path.join(baseFolder, sample, "01_intcorrect_med")
    if newDiskfolder is not False:
        targetFolder_med = os.path.join(newDiskfolder, sample, "01_intcorrect_med")
    if not os.path.exists(targetFolder_med):
        os.makedirs(targetFolder_med)
    
    if not parallel:
        for i in range(2016):
#            print(i)
            corefunction(i,sourceFolder,targetFolder_med)
                 
            
    elif parallel:
        corefunction(0,sourceFolder,targetFolder_med)
        result=Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(corefunction)(i,sourceFolder,targetFolder_med) for i in range(1,2016))
    
       
        
samples=makesamplelist(baseFolder)
length=len(samples)
c=1
#if parallel:
#    num_cores = mp.cpu_count()
#    print('Parallel processing of ',str(length),' samples on ',str(num_cores),' cores. Be patient...')
#    results=Parallel(n_jobs=num_cores)(delayed(correctIntensityProfile)(sample) for sample in samples)
#else:
for sample in samples:
    print(sample,'(',c,'/',length,')')
    c=c+1
    if sample[1]=='4': continue
    if sample in excluded_samples: continue
    # if not sample in repeats: continue
    correctIntensityProfile(sample)            
    if c  == 2:
        print("The water's changed to sand")
    if c == 4:
        print('Lakes and rivers turned to land')
    if c == 6:
        print("Plough up the rocky seas")
    if c == 8:
        print("Ride felled down trees")
    if c == 10:
        print("Foot by foot we edge")
    if c == 12:
        print("Once a ship, now a sled") 
    if c == 14:
        print("The water's changed to sand") 
    if c == 16:
        print("Lakes and rivers turned to land") #credit to Turisas (A Portage To the Unknown)    
print('done')      
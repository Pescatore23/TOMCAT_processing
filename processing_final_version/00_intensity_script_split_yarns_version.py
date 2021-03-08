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
# baseFolder = r"Z:\Robert_TOMCAT_3"
# baseFolder = r"F:\Zwischenlager_Robert\TOMCAT_3"
baseFolder = "E:\Robert_TOMCAT_5"

repeats = robpylib.TOMCAT.INFO.samples_to_repeat
#newDiskfolder=r'F:\Zwischenlager_Robert\TOMCAT_3'
# newDiskfolder=r'X:\TOMCAT3_processing_1'
# newBaseFolder=r'Y:\TOMCAT_3'
newDiskfolder = "E:\Robert_TOMCAT_5_split"

coordinates = {}
coordinates['T5_100_01'] = {}
coordinates['T5_100_01']['yarn1'] = [152, 152+236, 8, 8+268] #X, X+width, Y, Y+height as obtained from ImageJ
coordinates['T5_100_01']['yarn2'] = [364, 364+246, 471, 471+265]
coordinates['T5_100_02'] = {}
coordinates['T5_100_02']['yarn1'] = [17, 17+342, 204, 204+342]
coordinates['T5_100_02']['yarn2'] = [476, 476+268, 391, 391, 391+178]
coordinates['T5_100_3_II'] = {}
coordinates['T5_100_3_II']['yarn1'] = [384, 384+188, 0, 247]
coordinates['T5_100_3_II']['yarn2'] = [302, 302+203, 547, 547+317]
coordinates['T5_100_04'] = {}
coordinates['T5_100_04']['yarn1'] = [8, 8+292, 442, 442+246]
coordinates['T5_100_04']['yarn2'] = [669, 669+291, 297, 297+209]




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

def corefunction(z,sourceFolder,targetFolder_med_1, targetFolder_med_2, sample):
    breakFlag1=test_recalculate(sourceFolder,targetFolder_med_1,z)
    breakFlag2=test_recalculate(sourceFolder,targetFolder_med_2,z)
    if not breakFlag1 and not breakFlag2:
        Tstack, names, scans = robpylib.CommonFunctions.ImportExport.OpenTimeStack(sourceFolder, z)
        Tstack=np.int32(Tstack)
        coord1 = coordinates[sample]['yarn1']
        if not breakFlag1:
            Tstack1 = Tstack[coord1[2]:coord1[3],coord1[0]:coord1[1],:]
            refmed = np.median(Tstack1[:,:,0])
            for j in range(Tstack.shape[2]):
                Tstack1[:,:,j]=Tstack1[:,:,j]-(np.median(Tstack1[:,:,j])-refmed)
            Tstack1[Tstack1<0]=0
            Tstack1 = np.uint16(Tstack1)
            robpylib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder_med_1,Tstack1,names, scans)
        coord2 = coordinates[sample]['yarn2']
        if not breakFlag2:
            Tstack2 = Tstack[coord2[2]:coord2[3],coord2[0]:coord2[1],:]
            refmed = np.median(Tstack2[:,:,0])
            for j in range(Tstack.shape[2]):
                Tstack2[:,:,j]=Tstack2[:,:,j]-(np.median(Tstack2[:,:,j])-refmed)
            Tstack2[Tstack2<0]=0
            Tstack2 = np.uint16(Tstack2)
            robpylib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder_med_2,Tstack2,names, scans)
                           
        # Tstackmed=Tstack
        # refmed=np.median(Tstack[:,:,0])
        # for j in range(Tstack.shape[2]):
        #     Tstackmed[:,:,j]=Tstackmed[:,:,j]-(np.median(Tstackmed[:,:,j])-refmed)
        # Tstackmed[Tstackmed<0]=0
        # Tstackmed=np.uint16(Tstackmed)            
        # robpylib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder_med,Tstackmed,names, scans)
# yarn 1

# yarn 2




def correctIntensityProfile(sample, newDiskfolder=newDiskfolder, parallel=parallel):
    sourceFolder = os.path.join(baseFolder, sample, "00_raw")  #do this before registration in the future (replace by "00_raw")
    yarn_1_folder = os.path.join(baseFolder, ''.join([sample,'_yarn_1']))
    yarn_2_folder = os.path.join(baseFolder, ''.join([sample,'_yarn_2']))
    if newDiskfolder is not False:
        yarn_1_folder = os.path.join(newDiskfolder, ''.join([sample,'_yarn_1']))
        yarn_2_folder = os.path.join(newDiskfolder, ''.join([sample,'_yarn_2']))
    targetFolder_med_1 = os.path.join(yarn_1_folder, "01_intcorrect_med")
    targetFolder_med_2 = os.path.join(yarn_2_folder, "01_intcorrect_med")
    if not os.path.exists(yarn_1_folder):
        os.makedirs(yarn_1_folder)
    if not os.path.exists(targetFolder_med_1):
        os.makedirs(targetFolder_med_1)
    if not os.path.exists(yarn_2_folder):
        os.makedirs(yarn_2_folder)
    if not os.path.exists(targetFolder_med_2):
        os.makedirs(targetFolder_med_2)
    
    if not parallel:
        for i in range(2016):
#            print(i)
            corefunction(i,sourceFolder,targetFolder_med_1, targetFolder_med_2, sample)
                 
            
    elif parallel:
        corefunction(0,sourceFolder,targetFolder_med_1, targetFolder_med_2, sample)
        result=Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(corefunction)(i,sourceFolder,targetFolder_med_1, targetFolder_med_2, sample) for i in range(1,2016))
    
       
        
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
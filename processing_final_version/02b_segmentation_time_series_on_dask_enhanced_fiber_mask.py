# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:05:43 2018

Phase separation of 4D-Xray data of wicking in yarns

Input:
    - registered raw 4D-dataset
    - fiber mask (e.g. by weka segmentation of dry sample)
    
Output
    - time resolved water geometry


@author: firo by adapting Marcelo's code
"""

import os
import imageio
import robpylib
import numpy as np
from scipy.ndimage import morphology
from skimage import io
from joblib import Parallel, delayed
import multiprocessing as mp


#baseFolder=r'U:\TOMCAT_3_segmentation'
baseFolder = r'X:\TOMCAT3_processing_1'
#baseFolder = r'Z:\Robert_TOMCAT_3'

#newBaseFolder=False
newBaseFolder = r'W:\TOMCAT_3_segmentation'

parallel=True
waterpos=1600

excluded_samples=[
 'T3_100_4',        #FOV moved during acquisition
 'T3_300_6',        # water only in little pores
 'T3_300_3_III',    #no water
 'T4_025_3_III',    #wet from start
 'T3_300_8',        #FOV moved
 'T3_025_7',        #water only in small pores
 'T3_025_4_III',    #little water, data set incomplete, missing time steps reconstructed at PSI-ra-cluster, but only little water -> not reasonable to process?!
 'T3_025_3_II',      #little, water wet with acetone
 'T4_025_5_III',    #wet from start
 ]


#def test_recalculate(sourceFolder,targetFolder,z,OverWrite=False):
#    breakFlag=False
#    first_scan=os.listdir(sourceFolder)[0]
#    if os.path.exists(targetFolder):
#        if len(os.listdir(targetFolder))>0:
#            last_scan=os.listdir(targetFolder)[0]
#            if not OverWrite:
#                if os.listdir(os.path.join(sourceFolder,first_scan))[z] in os.listdir(os.path.join(targetFolder,last_scan)):
#                    breakFlag=True
#    return breakFlag

#def test_recalculate(sourceFolder,targetFolder,z,OverWrite=False):
#    breakFlag=False
#    first_scan=os.listdir(sourceFolder)[0]
#    if os.path.exists(targetFolder):   
#        if len(os.listdir(targetFolder))>0:
#            last_scan=os.listdir(targetFolder)[-1]
#            lst_nr=last_scan[-5:]
#            test_name=os.listdir(os.path.join(sourceFolder,first_scan))[z]
#            test_name=list(test_name)
#            test_name[-24:-19]=lst_nr
#            test_name=''.join(test_name)
#            if not OverWrite:
#                if test_name in os.listdir(os.path.join(targetFolder,last_scan)):
#                    breakFlag=True
#    return breakFlag

def test_recalculate(sourceFolder,targetFolder,z,OverWrite=False):
    breakFlag=False
    first_scan=os.listdir(sourceFolder)[0]
    content = os.listdir(os.path.join(sourceFolder, first_scan))
    if 'Thumbs.db' in content: content.remove('Thumbs.db')
    content.sort()
    sample_name = content[z]
    if os.path.exists(targetFolder):   
        if sample_name in os.listdir(targetFolder):
            breakFlag = True
    return breakFlag

def makesamplelist(baseFolder, **kwargs):
#    This function is a stub, expand it to apply sophisticated sample list creation on all processing steps
    samples=os.listdir(baseFolder)
#    samples=["32_200_025H2_cont"]
    return samples

def masking(Tstack, maskingthreshold=11000):
#    zero outside yarn, one inside
    shp=np.shape(Tstack)
    fake=np.zeros([shp[0],shp[1]],dtype=np.uint16)
    fake=Tstack[:,:,shp[2]-1]
    mask=fake.copy()
    mask[mask<maskingthreshold]=0
    mask[mask>0]=1
#    mask=mask-1
    mask=morphology.binary_fill_holes(mask)
#    masked=np.uint8(mask)
    return mask    


def get_jump_height(currpx,pos,pos2=0,receding=False):
    if not receding:
        low = np.median(currpx[max(0,pos-20):pos-2])              #"pos-20" changed to 10 because my time resolution is much coarser than Marcelo's
        hig = np.median(currpx[pos+2:min(len(currpx),pos+20)])
        jump = hig-low
    if receding:
        hig = np.median(currpx[max(pos,pos2-20):pos-2])
        low = np.median(currpx[pos2+2:min(len(currpx),pos2+20)])
        jump = low-hig
    return jump


def fft_grad_segmentation(imgs, poremask,z, waterpos=waterpos):
    check=4000
    if z<waterpos: check=9500
#    timg = np.zeros(np.shape(imgs), dtype='uint8')
    transitions = np.zeros([np.shape(imgs)[0],np.shape(imgs)[1]], dtype=np.uint16)
    transitions2 = transitions.copy()

#    tinitial = 10000 # initial threshold    -> deprecated (fibers are masked out before)
    jumpmin = 500.0 # minimum jumpheight, very low to capture corner flow, check if this causes add. noise
    X=Y=np.shape(imgs)[0]
    for pX in range(X):
        for pY in range(Y):
#            a=1
#            check if pixel is actually in the relevant pore space
            if poremask[pX,pY] >0:
                # current pixel time series
                currpx = imgs[pX,pY,:].astype(dtype='float')                
                s_filtered = robpylib.CommonFunctions.Tools.fourier_filter(currpx,band=0.1,dt=1.2)
                
#                # find where maximum gradient occurs
                g_filtered=np.gradient(s_filtered)

                pos = np.argmax(g_filtered)              
                jump = get_jump_height(currpx,pos)
    #            
                pos2 = np.argmin(g_filtered)
                jump2 = get_jump_height(currpx,pos,pos2=pos2,receding=True)
                
                if jump > jumpmin: # there is a transition; careful, water can also receed!! (can clearly be seen in rare cases)
#                    timg[pX,pY,:pos] = 0
#                    double check for noise
                    if np.median(currpx[min(pos+10,len(currpx)):min(pos+25,len(currpx))])>check:#7500:#9500:
#                    if np.median(currpx[min(pos+5,len(currpx)):min(pos+10,len(currpx))])>9500:
#                        timg[pX,pY,pos:] = 255
                        transitions[pX,pY] = pos
#                    else: timg[pX,pY,pos:] = 0
                    
                if jump2 < -2000 and pos2>pos and z<waterpos:
#                    timg[pX,pY,pos2:] = 0
                    transitions2[pX,pY] = pos2
    return transitions, transitions2


def core_function(z,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos):
    breakFlag=test_recalculate(sourceFolder,targetFolder_transitions,z,OverWrite=False)
    if not breakFlag:
        Tstack, names, scans = robpylib.CommonFunctions.ImportExport.OpenTimeStack(sourceFolder, z)
        fibername=fibernames[z]
        fibermask=np.uint8(io.imread(os.path.join(baseFolder,fibermaskFolder,fibername)))
        fibermask=fibermask/np.max(fibermask)
        mask=masking(Tstack)
        poremask=mask*(1-fibermask)
        Tstack=Tstack*poremask[:,:,None]
        transitions, transitions2 = fft_grad_segmentation(Tstack,poremask,z,waterpos=waterpos)
#        RobPyLib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder, binStack, names, scans)
        imageio.imwrite(os.path.join(targetFolder_transitions,names[0]),transitions)
        imageio.imwrite(os.path.join(targetFolder_transitions2,names[0]),transitions2)
    return True
        
        
        

def inner_segmentation_function(sample, newBaseFolder=False, tracefits=False, waterpos=waterpos):
    if not newBaseFolder: newBaseFolder = baseFolder
    sourceFolder = os.path.join(baseFolder, sample, '02_pystack_registered_from_5')#"02_registered_1300_rigid") 
    fibermaskFolder = os.path.join(baseFolder, sample, "01b_weka_segmented_mean","classified")   #has to be True for fibers and False for the rest, but not necessarly binary
    targetFolder = os.path.join(newBaseFolder, sample, "03_b_gradient_filtered_enhanced")   #FIXME: not used, but kept to minimize modification of code
    targetFolder_transitions = os.path.join(newBaseFolder, sample, "03_b_gradient_filtered_transitions_enhanced")
    targetFolder_transitions2 = os.path.join(newBaseFolder, sample, "03_b_gradient_filtered_transitions2_enhanced")
#       
#    if not os.path.exists(targetFolder):
#           os.makedirs(targetFolder) 
    if not os.path.exists(targetFolder_transitions):
           os.makedirs(targetFolder_transitions)
    if not os.path.exists(targetFolder_transitions2):
           os.makedirs(targetFolder_transitions2)

    zmax=2016
    
    fibernames=os.listdir(fibermaskFolder)
    if 'Thumbs.db' in fibernames: fibernames.remove('Thumbs.db')
    fibernames.sort()
    if sample[1]=='3': waterpos=1600
    
    if parallel:
        num_cores=mp.cpu_count()
        core_function(0,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos)
        Parallel(n_jobs=num_cores)(delayed(core_function)(z,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos) for z in range(1,zmax))
    else:
        for z in range(zmax):
            core_function(z,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos)
    return sample
  

def fft_segmentation(baseFolder=baseFolder, newDiskfolder=False):
    samples=makesamplelist(baseFolder)
    c=1
    for sample in samples:
        if not sample in robpylib.TOMCAT.INFO.samples_to_repeat: continue
        if not sample == 'T3_025_9_III': continue
        print(sample,'(',c,'/',len(samples),')')
        if sample[1]=='4':
            c=c+1
            continue
        inner_segmentation_function(sample, newBaseFolder=newBaseFolder)

        if c==1:
            print("Harvest the fields of time")
        if c==2:
            print("With the old man's scythe")
        if c==3:
            print("The narrow path of the chosen one")
        if c==4:
            print("Reaches beyond life")
        if c==5:
            print("I set sails for the ageless winds")
        if c==6:
            print("No fear of dying or a thought of surrender")
        if c==7:
            print("I threaten every barrier on my way")
        if c==8:
            print("I am bound forever with token of time")
        if c==9:
            print("Among the humble people")
        if c==10:
            print("Everything is torn apart")
        if c==11:
            print("But I'm blessed with faith")
        if c==12:
            print("And bravely I shall go on")
        if c==13:
            print("I set sails for the ageless winds")
        if c==14:
            print("No fear of dying or a thought of surrender")
        if c==15:
            print("I threaten every barrier on my way")
        if c==16:
            print("I am bound forever with token of time")
        if c==17:
            print("Are thou the bringer of hope and joy")
        if c==18:
            print("That I've waited for years, I shall fight to restore the moon")
        if c==19:
            print("Wisdoms of time are carved on the sacred wood")
        if c==20:
            print("Do thou possess spiritual powers")
        if c==21:
            print("That would dispel all my fears, I shall not die until the seal is broken")
        if c==22:
            print("Token of time is trusted in the hands of the chosen one")
        if c==23:
            print("I set sails for the ageless winds")
        if c==24:
            print("No fear of dying or a thought of surrender")
        if c==25:
            print("I threaten every barrier on my way")
        if c==26:
            print("I am bound forever with token of time")
        c=c+1 
#            #credits to Ensiferum (Token of Time)


    return
             
         
results=fft_segmentation(baseFolder=baseFolder)      
print('done')      
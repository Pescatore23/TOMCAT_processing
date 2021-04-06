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
# import multiprocessing as mp
# from skimage.morphology import square
from skimage.morphology import convex_hull_image
import socket
host = socket.gethostname()
repeats = robpylib.TOMCAT.INFO.samples_to_repeat

#baseFolder=r'U:\TOMCAT_3_segmentation'
# baseFolder = r'O:\TOMCAT3_processing_1'
# baseFolder = r'F:\Zwischenlager_Robert\TOMCAT_3'
# baseFolder = r'E:\Robert_TOMCAT_3b'
baseFolder = r"E:\Robert_TOMCAT_5_split"

OverWrite = True
newBaseFolder=False
# newBaseFolder = r'W:\TOMCAT_3_segmentation'
num_cores = 32
temp_folder= r"Z:\users\firo\joblib_tmp"
if host == 'ddm05307':
    pc = True
    num_cores = 16
    temp_folder = None
parallel=True
waterpos=2016

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
  'T5_100_02_yarn_2'  #some strange error
 ]


def test_recalculate(sourceFolder, targetFolder, z, OverWrite = OverWrite):
    breakFlag = False
    if not OverWrite:
        first_scan=os.listdir(sourceFolder)[0]
        if os.path.exists(targetFolder):
            if len(os.listdir(targetFolder))>0:
                    test_name=os.listdir(os.path.join(sourceFolder,first_scan))[z]
                    if test_name in os.listdir(targetFolder):
                        breakFlag = True
    return breakFlag

def makesamplelist(baseFolder, **kwargs):
#    This function is a stub, expand it to apply sophisticated sample list creation on all processing steps
    samples=os.listdir(baseFolder)
#    samples=["32_200_025H2_cont"]
    return samples

def masking(Tstack, maskingthreshold=25000):  #26000 for 3b, 110000 for 3
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

def interlace_masking(Stack, maskingthreshold=11000):
    ref_mean = Stack[:,:,:1000].mean()
    for z in range(Stack.shape[2]):
        Stack[:,:,z] = Stack[:,:,z] - (Stack[:,:,z].mean() - ref_mean)
    Stack[np.where(Stack<0)] = 0
    Stack[Stack<maskingthreshold]=0
    Stack[Stack>0]=1
    return Stack


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
    # check=6000 #T3,T4
    check = 15000 #T5
    # check = 20000 #T2
    # if z<waterpos: check=9500
    if z<waterpos: check=25000
    # timg = np.zeros(np.shape(imgs), dtype='uint8')
    transitions = np.zeros([np.shape(imgs)[0],np.shape(imgs)[1]], dtype=np.uint16)
    transitions2 = transitions.copy()

#    tinitial = 10000 # initial threshold    -> deprecated (fibers are masked out before)
    jumpmin = 1500.0 # minimum jumpheight
    X=imgs.shape[0]
    Y=imgs.shape[1]
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

                pos = np.argmax(g_filtered)-1                
                jump = get_jump_height(currpx,pos)
    #            
                pos2 = np.argmin(g_filtered)-1
                jump2 = get_jump_height(currpx,pos,pos2=pos2,receding=True)
                
                if jump > jumpmin: # there is a transition; careful, water can also receed!! (can clearly be seen in rare cases)
                    # timg[pX,pY,:pos] = 0
#                    double check for noise
                    if np.median(currpx[min(pos+10,len(currpx)):min(pos+25,len(currpx))])>check:#7500:#9500:
#                    if np.median(currpx[min(pos+5,len(currpx)):min(pos+10,len(currpx))])>9500:
                        # timg[pX,pY,pos:] = 255
                        transitions[pX,pY] = pos
                    # else: timg[pX,pY,pos:] = 0
                    
                if jump2 < -2000 and pos2>pos and z<waterpos:
                    # timg[pX,pY,pos2:] = 0
                    transitions2[pX,pY] = pos2
    return transitions, transitions2 #, timg 


def core_function(z,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos, mask=None):
    breakFlag=test_recalculate(sourceFolder,targetFolder_transitions,z,OverWrite=OverWrite)
    if not breakFlag:
        Tstack, names, scans = robpylib.CommonFunctions.ImportExport.OpenTimeStack(sourceFolder, z)
        fibername=fibernames[z]
        fibermask=np.uint8(io.imread(os.path.join(baseFolder,fibermaskFolder,fibername)))
        fibermask=fibermask/np.max(fibermask)
        hullmask = convex_hull_image(fibermask>0).astype(np.uint8) #hullmask introduced for TOMCAT_3b
        
        # if fibername[1]=='3':
        # mask=masking(Tstack)
            
        # poremask=hullmask*mask*(1-fibermask)
        poremask=hullmask*(1-fibermask) #
        Tstack=Tstack*poremask[:,:,None] #hullmask might be more computational effort, but seems more reliable
        
        # binStack, transitions, transitions2 = fft_grad_segmentation(Tstack,poremask,z,waterpos=waterpos)
        transitions, transitions2 = fft_grad_segmentation(Tstack,poremask,z,waterpos=waterpos)
        # robpylib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder, binStack, names, scans)
        imageio.imwrite(os.path.join(targetFolder_transitions,names[0]),transitions)
        imageio.imwrite(os.path.join(targetFolder_transitions2,names[0]),transitions2)
    return True
        
        
        

def inner_segmentation_function(sample, newBaseFolder=False, tracefits=False, waterpos=waterpos):
    if not newBaseFolder: newBaseFolder = baseFolder
    sourceFolder = os.path.join(baseFolder, sample, '02_pystack_registered')#"02_registered_1300_rigid")
    # sourceFolder = os.path.join(baseFolder, sample, '02_pystack_registered_from_5')#"02_registered_1300_rigid") 
    fibermaskFolder = os.path.join(baseFolder, sample, "01a_weka_segmented_dry","classified")   #has to be True for fibers and False for the rest, but not necessarly binary
    targetFolder = os.path.join(newBaseFolder, sample, "03_gradient_filtered")
    targetFolder_transitions = os.path.join(newBaseFolder, sample, "03_gradient_filtered_transitions")
    targetFolder_transitions2 = os.path.join(newBaseFolder, sample, "03_gradient_filtered_transitions2")
#       
    # if not os.path.exists(targetFolder):
           # os.makedirs(targetFolder) 
    if not os.path.exists(targetFolder_transitions):
           os.makedirs(targetFolder_transitions)
    if not os.path.exists(targetFolder_transitions2):
           os.makedirs(targetFolder_transitions2)

    zmax=2016
    
    fibernames=os.listdir(fibermaskFolder)
    if 'Thumbs.db' in fibernames: fibernames.remove('Thumbs.db')
    fibernames.sort()
    if sample[1]=='3': 
        waterpos=1600
    if sample[1]=='4':
        last_scan = os.path.join(sourceFolder, os.listdir(sourceFolder)[-1])
        masks, _  = robpylib.CommonFunctions.ImportExport.ReadStackNew(last_scan)
        masks = interlace_masking(masks)
    if sample[1]=='_':
        last_scan = os.path.join(sourceFolder, os.listdir(sourceFolder)[-1])
        masks, _  = robpylib.CommonFunctions.ImportExport.ReadStackNew(last_scan)
        masks = interlace_masking(masks, maskingthreshold=25000)  
    if parallel:
        # num_cores=mp.cpu_count()
       
        if sample[1] == '4' or sample[1]=='_':
            core_function(0,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos, mask=masks[:,:,0])
            Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(core_function)(z,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos, mask=masks[:,:,z]) for z in range(1,zmax))
        else:
            core_function(0,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos)
            Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(core_function)(z,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos) for z in range(1,zmax))
    # else:
        # for z in range(zmax):
            # core_function(z,fibermaskFolder,sourceFolder,targetFolder,targetFolder_transitions,targetFolder_transitions2,fibernames,waterpos=waterpos)
    return sample
  

def fft_segmentation(baseFolder=baseFolder, newDiskfolder=False):
    samples=makesamplelist(baseFolder)
    c=1
    for sample in samples:
        if sample in excluded_samples: continue
#        if not sample == 'T3_025_9_III': continue
        # if not sample in repeats: continue
        # if sample[1] == '4': continue
        if sample == 'T3_025_1': continue
        # if sample in robpylib.TOMCAT.INFO.interlace_good_samples: continue
        print(sample,'(',c,'/',len(samples),')')
#        if sample[1]=='4':
#            c=c+1
#            continue
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
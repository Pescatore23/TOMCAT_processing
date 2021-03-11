# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:19:32 2019

@author: firo
"""
import os
import numpy as np
import robpylib
from pystackreg import StackReg
from skimage import transform as tf
import random
from joblib import Parallel, delayed
# import multiprocessing as mp

num_cores = 32
temp_folder = r"Z:\users\firo\joblib_tmp"
waterpos=2000

parallel=True

number_reference_slices=100
mount=''
# baseFolder=r'Z:\Robert_TOMCAT_3'
# baseFolder = r'I:\disk1'
#baseFolder=r'O:\disk2'
#baseFolder=r'T:\TOMCAT3_Test'
#baseFolder = r'S:\Zwischenlager\disk1'
#baseFolder=r'Y:\TOMCAT_3'
# baseFolder = r"E:\Robert_TOMCAT_5_split"
baseFolder = r"E:\Robert\Robert_TOMCAT_2"

#newBaseFolder=r'X:\TOMCAT3_processing_1'
#newBaseFolder=r'Y:\TOMCAT_3'
newBaseFolder = baseFolder


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

repeats = robpylib.TOMCAT.INFO.samples_to_repeat

def makesamplelist(baseFolder, **kwargs):
    samples=os.listdir(baseFolder)
    
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


# Tstack needs to be in shape [t,x,y], use shape='t_x' for reading and writing Tstack
# pystackreg wqorks with frame as the first index and also transformation matrices are stored this way    

def get_transformation(Tstack):
#    Tstack=np.transpose(Tstack,(2,0,1))
    sr = StackReg(StackReg.RIGID_BODY)
    trans_matrix= sr.register_stack(Tstack, reference='first')
    # trans_matrix= sr.register_stack(Tstack, reference='previous')
    outStack = sr.transform_stack(Tstack)   
#    outStack = np.transpose(outStack,(1,2,0))
    outStack=np.uint16(outStack)
    return outStack, trans_matrix

def apply_transformation(Tstack, trans_matrix):
#    Tstack=np.transpose(Tstack,(2,0,1))
    outStack=np.zeros(Tstack.shape).astype(np.float)
    for t in range(Tstack.shape[0]):
#        print('t = ',t)
        tform = tf.AffineTransform(matrix=trans_matrix[t,:,:])
        outStack[t,:,:]=tf.warp(Tstack[t,:,:],tform)
#    outStack=np.transpose(outStack,(1,2,0))
    outStack=np.uint16(outStack*2**16)
    return outStack

def merge_transformation_matrices(matFolder):
    names=os.listdir(matFolder)
    if 'Thumbs.db' in names: names.remove('Thumbs.db')
    test=np.load(os.path.join(matFolder,names[0]))
    mats=np.zeros([len(names),test.shape[0],test.shape[1],test.shape[2]])
    for z in range(len(names)):
        mats[z,:,:,:]=np.load(os.path.join(matFolder,names[z]))
    res_trans_mat=np.mean(mats, axis=0)
    return res_trans_mat


def register_slice(sourceFolder,targetFolder,matFolder,z,regfile, sample=None, trans_mat_flag=False,trans_mat=False):
    breakflag = test_recalculate(sourceFolder,targetFolder,z)
    if not breakflag:
        try:
            Tstack,names,scans=robpylib.CommonFunctions.ImportExport.OpenTimeStack(sourceFolder,z, orient='t_x')
#        print(names[0])
            
            if sample in robpylib.TOMCAT.INFO.samples_to_repeat:
                Tstack = Tstack[4:,:,:]
                names = names[4:]
                scans = scans[4:]
            
            if trans_mat_flag:
                outStack=apply_transformation(Tstack, trans_mat)
                robpylib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder, outStack, names, scans, orient='t_x')
            else:
                outStack, trans_mat = get_transformation(Tstack)
                np.save(os.path.join(matFolder,''.join(['trans_mat_z_',str(z),'.npy'])),trans_mat)
                robpylib.CommonFunctions.ImportExport.WriteTimeSeries(targetFolder, outStack, names, scans, orient='t_x')
        except:
            reglog=open(''.join([regfile,str(z),'.tif']),"w")
            reglog.write('\n'+'slice '+str(z)+' failed')
            reglog.close()
    return True
        

def register(sample, baseFolder=baseFolder, newBaseFolder=False, stage='00_raw', num_cores=num_cores, waterpos=waterpos, parallel=parallel):
    zmax=2016
    sourceFolder = os.path.join(baseFolder, sample, stage)
    # targetFolder = os.path.join(baseFolder, sample, '02_pystack_registered_prev_ref')
    # matFolder = os.path.join(baseFolder, sample,'02_pystack_matrices_prev_ref')
    targetFolder = os.path.join(baseFolder, sample, '02_pystack_registered')
    matFolder = os.path.join(baseFolder, sample,'02_pystack_matrices')
    
    if newBaseFolder is not False:
        # targetFolder = os.path.join(newBaseFolder, sample, '02_pystack_registered_prev_ref')
        # matFolder = os.path.join(newBaseFolder, sample,'02_pystack_matrices_prev_ref')
        targetFolder = os.path.join(newBaseFolder, sample, '02_pystack_registered')
        matFolder = os.path.join(newBaseFolder, sample,'02_pystack_matrices')
        
    if not os.path.exists(newBaseFolder):
        os.mkdir(newBaseFolder)
    if not os.path.exists(os.path.join(newBaseFolder,sample)):
        os.mkdir(os.path.join(newBaseFolder,sample))
            
    if not os.path.exists(targetFolder):
        os.mkdir(targetFolder)
    if not os.path.exists(matFolder):
        os.mkdir(matFolder)
        
    if sample[1]=='3': waterpos=1500
#    get transformation for random slices
    print('get trans mat')
    slicelist=random.sample(range(waterpos),number_reference_slices)    
    regfile=os.path.join(baseFolder,sample,'register_log.txt')
    reglog=open(regfile,"w")
    reglog.write(str(number_reference_slices)+' slices registered: '+str(slicelist)+'\n')
    reglog.write('Water position given as slice '+str(waterpos))
    reglog.close() 

    register_slice(sourceFolder,targetFolder,matFolder,slicelist[0],regfile)
    if parallel:
        if sample[1]=='3':          #register every slice separately in the case of single yarns
            result=Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(register_slice)(sourceFolder,targetFolder,matFolder,z,regfile, sample=sample ) for z in range(zmax))
        
        else:                       #interlaces seem to be suitable to save computation time by just registering some test slices and then apply transformation onto the rest
            result=Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(register_slice)(sourceFolder,targetFolder,matFolder,z,regfile, sample = sample) for z in slicelist[1:])
            print('apply trans mat')            
        #            apply transformation on rest
            trans_mat=merge_transformation_matrices(matFolder)
            result=Parallel(n_jobs=num_cores, temp_folder=temp_folder)(delayed(register_slice)(sourceFolder,targetFolder,matFolder,z,regfile, sample=sample, trans_mat_flag=True,trans_mat=trans_mat) for z in range(zmax))
    else:
        for z in slicelist[1:]:
            register_slice(sourceFolder,targetFolder,matFolder,z)
        trans_mat=merge_transformation_matrices(matFolder)
        print('apply trans mat')
        for z in range(zmax):
            try:
                register_slice(sourceFolder,targetFolder,matFolder,z,regfile, sample= sample, trans_mat_flag=True,trans_mat=trans_mat)
            except:
                print('slice ',z,' of sample ',sample,' failed!')
#    for z in range(2016):
#        print(z)
#        register_slice(sourceFolder,targetFolder,matFolder,z,trans_mat_flag=True,trans_mat=trans_mat)


"""
___________________
"""

c=0
samples=os.listdir(baseFolder)


print(len(samples),' samples to calculate')

for sample in samples:
    if sample in excluded_samples: continue
    # if not sample == 'T4_025_2_II': continue
    # if not sample in repeats: continue
#    if sample[1] == '4': continue
#    if not sample == 'T4_300_5_III': continue
    # if sample == 'T4_025_4': continue #recos incomplete
    folder=baseFolder
    newFolder=newBaseFolder
    stage='00_raw'   
#        Straight yarn samples were corrected for their intensity in a first step and already tansferred to new location
    if sample[1]=='3':
#        continue
        stage='01_intcorrect_med'
        # folder=newBaseFolder
        # newFolder=False
    if sample[1]=='5':
#        continue
        stage='01_intcorrect_med'
        # folder=newBaseFolder
        # newFolder=False       
    if sample[0]=='Y': continue

    c=c+1   
    print(sample,'(',c,'/',len(samples),')')
    register(sample,baseFolder=folder,newBaseFolder=newFolder,stage=stage)
    if c==1:
        print("So mysterious is your world,")
    if c==2:
        print("Concealed beyond the stars")
    if c==3:
        print("Far away from the earth,")
    if c==4:
        print("It flows one with time and dark as the night")
    if c==5:
        print("Million shapes and colors")
    if c==6:
        print("Are storming inside your mind")
    if c==7:
        print("Creating endless dimensions")
    if c==8:
        print("Forming universes without walls")
    if c==9:
        print("Let go of the stars, the stars that fell into the sea")
    if c==10:
        print("Let go of your thoughts and dreams,")
    if c==11:
        print("What can you see now")
    if c==12:
        print("Wanderer of time")
    if c==13:
        print("It's too late now")
    if c==14:
        print("Creator of Dimensions")
    if c==15:
        print("Destroy the walls of time")
    if c==16:
        print("Hands of the blind are holding your fate")
    if c==17:
        print("Tides of life will take you away, will take you away")
    if c==18:
        print("Starchild!")
    if c==19:
        print("Visions are born from the unknown force")
    if c==20:
        print("It dominates the way of time")
    if c==21:
        print("The dream only ends, when the worlds come to an end,")
    if c==22:
        print("Starchild!")
    if c==23:
        print("You cannot escape to the dark streams of the sea,")
    if c==24:
        print("To suppress your dreams")
    if c==25:
        print("Nothing can keep you away from the need to create")
    if c==26:
        print("Cause your path is free")
#        credit to Wintersun (Starchild, 2004)
    
        
        
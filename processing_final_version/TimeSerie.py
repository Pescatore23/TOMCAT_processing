

import sys
import os

homeCodePath= "H:\\10_Python\\005_Scripts_from_others\\005_Marcelos_Processing_Files\\wicking-yarn-master\\wicking-yarn-master"
if homeCodePath not in sys.path:
    sys.path.append(homeCodePath)

from ij import IJ

import myFunctions as mf

slice_number=400
sample="R_m4_33_050_2"

processing_stage='02_pystack_registered_leg_0'
#processing_stage='00_raw'
#processing_stage='03_gradient_filtered'
#processing_stage = '03_reg_ML_seg'

#baseFolder = "V:\\TOMCAT_II_1"
#baseFolder = "U:\\disk1"
#baseFolder= r"W:\TOMCAT_3_segmentation"
#baseFolder = r'E:\\Robert_TOMCAT_3b'
#baseFolder = r"Z:\Robert_TOMCAT_4"
#baseFolder =r"E:\Robert_TOMCAT_3b"
baseFolder = r"N:\Dep305\Robert_Fischer\Robert_TOMCAT_2"

samples = os.listdir(baseFolder)
samples = [sample]

for sample in samples:
	sourceFolder = os.path.join(baseFolder, sample, processing_stage)
	original, names, scans = mf.openSampleImages(sourceFolder, slice_number, name=''.join([sample,"_slice_",str(slice_number)]))
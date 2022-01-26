"""
Classify TOMCAT data with weka segmentation

run inside fiji
, somehow you have to open the weka plugin once to make the module available, maybe find a way around

before adding more fancy features, try the first five (simple) ones, they are usually a good bet; others can confuse the AI

created by: Robert Fischer
18.07.2018
"""

import sys
#MarceloCodePath= "H:\\10_Python\\005_Scripts_from_others\\005_Marcelos_Processing_Files\\wicking-yarn-master\\wicking-yarn-master"
#homeCodePath=r'F:\Zwischenlager_Robert'
homeCodePath=r"H:\10_Python\008_TOMCAT_processing\tomcat-processing\processing_final_version"
#if MarceloCodePath not in sys.path:
#    sys.path.append(MarceloCodePath)
if homeCodePath not in sys.path:
	sys.path.append(homeCodePath)

from ij import IJ
import trainableSegmentation
import os
#import myFunctions as mf
import time
import RobertFijiFunctions as rff
from ij.io import FileSaver


baseFolder = r"N:\Dep305\Robert_Fischer\Robert_TOMCAT_2"

sample = 'R_m4_33_050_2'

leg = '_leg_1'

sourceFolder = os.path.join(baseFolder,sample,''.join(['01_intcorrect_med', leg]))

targetFolder = os.path.join(baseFolder,sample,''.join(["04_weka_segmented",leg])) 

if not os.path.exists(targetFolder):
	os.makedirs(targetFolder)

timestepnames = os.listdir(sourceFolder)
print(sample,leg)

#segmentator = trainableSegmentation.WekaSegmentation()
#segmentator.loadClassifier(r"H:\\11_Essential_Data\\03_TOMCAT\\11_thin_interlaces\\R_m4_33_050_2\\weka\\classifier\\leg_0_slice_400_time_series_spatial_dry_200time_leg_2_refine.model")	
#print('model loaded')
for timestep in timestepnames[2:]:

	print(timestep)
	
	timestep_folder = os.path.join(sourceFolder, timestep)
	target = os.path.join(targetFolder, timestep)

	if not os.path.exists(target):
		os.makedirs(target)

	test_set, _ = rff.openSilentStack(timestep_folder)

	segmentator = trainableSegmentation.WekaSegmentation(test_set)
	segmentator.loadClassifier(r"H:\\11_Essential_Data\\03_TOMCAT\\11_thin_interlaces\\R_m4_33_050_2\\weka\\classifier\\leg_0_slice_400_time_series_spatial_dry_200time_leg_2_refine.model")	
	print('model loaded')
	numThreads=0 #0 is autodetected
	
	result = segmentator.applyClassifier(test_set, 0, 0)     #0 for labeled image, 1 for probability map of each phase (=hyperstack)

	#result = segmentator.getClassifiedImage()
	if not os.path.exists(os.path.join(target,'temp')):
			  os.makedirs(os.path.join(target,'temp'))
	FileSaver(result).saveAsTiff(''.join([target,'\\temp\\wekastack.tif']))
	
	test_set = None
	

	#for now, convert the tiff-Satck to tiff-sequence manually afterwards (does take less than fighting jython, fiji and weka, is also more efficnet due to parellel computing inside weka segmentation)
	#maybe do a macro opening the stack and save it as sequence; probablity image is stored, to get binary just thershold >0.5 or anything reasonable
	
	IJ.run("Close All")



"""
Classify TOMCAT data with weka segmentation

run inside fiji
, somehow you have to open the weka plugin once to make the module available, maybe find a way around

before adding more fancy features, try the first five (simple) ones, they are usually a good bet; others can confuse the AI

created by: Robert Fischer
18.07.2018
"""

from ij import IJ, ImagePlus, ImageStack
import trainableSegmentation
import os
#import myFunctions as mf
import time
from ij.io import FileSaver

def openSilentStack(folder, show=False, name="stack"):
	imlist=os.listdir(folder)
	imlist.sort()
	isFirst = True
	names = []
	
	for im in imlist:
		if im == "Thumbs.db": continue
		currImp = IJ.openImage(folder+"/"+im)
		names.append(im)
		
		if isFirst:
			stack= ImageStack(currImp.getWidth(), currImp.getHeight())
			isFirst = False
		stack.addSlice(currImp.getProcessor())
		
	imp = ImagePlus("stack",stack)
	if show == True:
		imp.show()
	imp.setTitle(name)
	return imp, names
#Interlace_training_path=r'R:\Scratch\305\_Robert\05_weka_training_sets\Interlaces'

repeats = ['T3_300_3', 'T3_025_1', 'T3_025_4', 'T3_025_9_III']

#baseFolder=r'F:\Zwischenlager_Robert\TOMCAT_3'
baseFolder=r"/Users/firo/NAS/Robert_TOMCAT_3_Part_2"
#baseFolder=r'T:\disk2'
#baseFolder=r'U:\TOMCAT_3_segmentation'
#baseFolder=r'V:\disk2'
#baseFolder = r'X:\TOMCAT3_processing_1'
#baseFolder = r'Z:\TOMCAT_3'

excluded_samples=[
 'T3_100_4',        #FOV moved during acquisition
 'T3_300_6',        # water only in little pores
 'T3_300_3_III',    #no water
 'T4_025_3_III',    #wet from start
 'T4_025_5_III',    # wet from start
 'T3_300_8',        #FOV moved
 'T3_025_7',        #water only in small pores
 'T3_025_4_III',    #little water, data set incomplete, missing time steps reconstructed at PSI-ra-cluster, but only little water -> not reasonable to process?!
 'T3_025_3_II'      #little water, wet with acetone
 ]    

def makesamplelist(baseFolder, **kwargs):
#    This function is a stub, expand it to apply sophisticated sample list creation on all processing steps
    samples=os.listdir(baseFolder)
    return samples


samples=makesamplelist(baseFolder)


time00=time.time()
#sourceFolder = '02_pystack_registered'
#sourceFolder = '00_raw'
sourceFolder = '02_pystack_registered_from_5'

Features = [
			[1   ,   'Gaussian_blur'],
			[1   ,   'Sobel_filter'],
			[1   ,   'Hessian'],
			[1   ,   'Difference_of_gaussians'],
			[1   ,   'Membrane_projections'],
			[0   ,   'Variance'],
			[0   ,   'Mean'],
			[0   ,   'Minimum'],
			[0   ,   'Maximum'],
			[0   ,   'Median'],
			[0   ,   'Anisotropic_diffusion'],
			[0   ,   'Bilateral'],
			[0   ,   'Lipschitz'],
			[0   ,   'Kuwahara'],
			[0   ,   'Gabor'],
			[0   ,   'Derivatives'],
			[0   ,   'Laplacian'],
			[0   ,   'Structure'],
			[0   ,   'Entropy'],
			[0  ,   'Neighbors']
			]

enableFeatures=list(list(zip(*Features))[0])
FeatureNames=list(list(zip(*Features))[1])
usedFeatures=[]
j=0
for i in enableFeatures:
	if i>0:
		usedFeatures.append(FeatureNames[j])
	j=j+1

i=1


for sample in reversed(samples):
	if not sample in repeats: continue
	print(sample, "(",i,"/",len(samples),")")
	i=i+1
	if sample[1] == '4': continue
	#if sample in excluded_samples: continue
	#if not sample == 'T3_025_3_III': continue
	
	interlaceFlag=False
	targetFolder = os.path.join(baseFolder,sample,"01a_weka_segmented_dry") 
#	targetFolder = os.path.join(baseFolder,sample,"01b_weka_segmented_mean") 



	if sample[1]=='4':
	   interlaceFlag=True
	   interlaceFlag=False
	   continue
	 #  training_path=Interlace_training_path
	#elif sample[1]=='3':
	#training_path=Yarn_training_path
	training_path=os.path.join(baseFolder, sample, targetFolder, "training_set")
	#else:
	 #  print('Sample ',sample,' is not a recognized experiment type')
	  # continue

	   
	fiberlabel_folder=os.path.join(training_path,"fiber_label")
	trainingset_folder=os.path.join(training_path, "training_raw")       
	time0=time.time()



	if not os.path.exists(targetFolder):
		os.mkdir(targetFolder)

	if os.path.exists(os.path.join(targetFolder,'temp','wekastack.tif')):
		print(sample, 'already classified')
		continue


	fiber_label, _ = openSilentStack( fiberlabel_folder )
	training_set, _ = openSilentStack( trainingset_folder )

	numThreads=0 #0 is autodetected

	segmentator = trainableSegmentation.WekaSegmentation( training_set )

	segmentator.addBinaryData(fiber_label, 0,"class 2", "class 1")   #True added to first string ("class 2"), False added to second string ("class1")

	segmentator.setEnabledFeatures( enableFeatures)
	segmentator.trainClassifier()


	dryscan=os.listdir(os.path.join(baseFolder, sample, sourceFolder))[0]
	folder=os.path.join(baseFolder, sample, sourceFolder, dryscan)
	folder2=''.join([folder,'_2'])
	folder3=''.join([folder,'_3'])

# SECOND VERSION USING THE TEMPORAL MEAN AS REFERENCE
#	folder=os.path.join(baseFolder, sample, '02a_temporal_mean')
#	folder2=''.join([folder,'_2'])
#	folder3=''.join([folder,'_3'])
	

	x_folder=os.path.join(baseFolder, sample, targetFolder, "x_set")
	y_folder=os.path.join(baseFolder, sample, targetFolder, "y_set")
	  
	  
	test_set, names = openSilentStack(folder)
	result = segmentator.applyClassifier(test_set, numThreads, 1)     #0 for labeled image, 1 for probability map of each phase (=hyperstack)

	if not os.path.exists(os.path.join(targetFolder,'temp')):
			  os.makedirs(os.path.join(targetFolder,'temp'))
	FileSaver(result).saveAsTiff(''.join([targetFolder,'\\temp\\wekastack.tif']))


	if os.path.exists(folder2):
		test_set, names = openSilentStack(folder2)
		result = segmentator.applyClassifier(test_set, numThreads, 1)     #0 for labeled image, 1 for probability map of each phase (=hyperstack)

		if not os.path.exists(os.path.join(targetFolder,'temp')):
			 	 os.makedirs(os.path.join(targetFolder,'temp'))
		FileSaver(result).saveAsTiff(''.join([targetFolder,'\\temp\\wekastack2.tif']))


	if os.path.exists(folder3):
		test_set, names = openSilentStack(folder3)
		result = segmentator.applyClassifier(test_set, numThreads, 1)     #0 for labeled image, 1 for probability map of each phase (=hyperstack)

		if not os.path.exists(os.path.join(targetFolder,'temp')):
			 	 os.makedirs(os.path.join(targetFolder,'temp'))
		FileSaver(result).saveAsTiff(''.join([targetFolder,'\\temp\\wekastack3.tif']))
	
	test_set = None
	if interlaceFlag:
		if not os.path.exists(os.path.join(targetFolder,'temp','x_set')):
				  os.makedirs(os.path.join(targetFolder,'temp','x_set'))
		if not os.path.exists(os.path.join(targetFolder,'temp','y_set')):
				  os.makedirs(os.path.join(targetFolder,'temp','y_set'))
		x_set , _ = openSilentStack(x_folder)
		x_result=segmentator.applyClassifier(x_set, numThreads, 1)
		FileSaver(x_result).saveAsTiff(''.join([targetFolder,'\\temp\\x_set\\wekastack.tif']))
		
		y_set , _ = openSilentStack(y_folder)
		y_result=segmentator.applyClassifier(y_set, numThreads, 1)
		FileSaver(y_result).saveAsTiff(''.join([targetFolder,'\\temp\\y_set\\wekastack.tif']))	



	tottime=time.time()-time00
	logfile=open(''.join([targetFolder,'\\temp\\wekalog.txt']),'w')
	for item in usedFeatures:
		logfile.write("%s\n" % item)
	logfile.write("%s\n" % tottime)
	logfile.close()

	#for now, convert the tiff-Satck to tiff-sequence manually afterwards (does take less than fighting jython, fiji and weka, is also more efficnet due to parellel computing inside weka segmentation)
	#maybe do a macro opening the stack and save it as sequence; probablity image is stored, to get binary just thershold >0.5 or anything reasonable

	IJ.run("Close All")



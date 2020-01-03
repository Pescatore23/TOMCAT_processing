from ij import IJ, ImagePlus, ImageStack
import os
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

	
	

##creates a list with all paths and names of the newest reconstrusction of samples with recorded water uptake	
#
#def newRecoList(mountlocation):
#    newRecoPath=''.join([mountlocation,"newReco"])   #mounlocation i.e. "T:\\" on windows or /mnt/t/  on linux or whatever
#    disks=os.listdir(newRecoPath)
#    newsamples=[]
#    for disk in disks:
#        newsamples.append(os.listdir(os.path.join(newRecoPath,disk)))
#    
#    newsamples2 = [item for sublist in newsamples for item in sublist]
#    newsamples2.append('Robert')
#    newsamples2.append('_hide')
#    newsamples2.append('10_050_025H2_wet')
#    newsamples2.append('10_200_300H2_01_b')
#    return newsamples2	
#	
#def makesamplefolderlist(mount):
#    samples=[]
#    
#    newsamples=newRecoList(mount)
#	
#    wetsamples= [
#	'03_500_100H1',
#	'10_050_025_H3',
#	'10_050_025H3',
#	'10_050_100_H2',
#	'10_500_100H1',
#	'10_500_300H1',
#	'32_050_025_H1_cont',
#	'32_050_300_H1',
#	'32_200_025H2_cont',
#	'32_200_300_H1',
#	'32_500_025_H1',
#	'32_500_300_H1',
#	'10_050_100H3',
#	'10_050_300H3',
#	'10_200_025H2',
#	'10_200_100H2',
#	'10_200_300H2',
#	'10_500_025H3',
#	'10_500_100H2',
#	'10_500_300H2']
#    
##    disk1
#    print('disk1')
#    foldercontent=os.listdir(os.path.join(mount,'disk1'))
#    
#    for sample in foldercontent:
#        if sample in newsamples: continue
#        
#    if sample in wetsamples:
#        path=os.path.join(mount,'disk1',sample,'00_raw')
#        samples.append([path,sample])
#        #print(sample, path)
#        
##    diks2
#    print('disk2')
#    foldercontent=os.listdir(os.path.join(mount,'disk2'))
#    for sample in foldercontent:
#        if sample in newsamples: continue
#    
#    if sample in wetsamples:
#        path=os.path.join(mount,'disk2',sample,'00_raw')
#        samples.append([path,sample])
#        #print(sample, path)
#        
##   newReco
#    print('newReco')
#    topfoldercontent=os.listdir(os.path.join(mount,'newReco'))
#    for folder in topfoldercontent:
#        foldercontent=os.listdir(os.path.join(mount,'newReco',folder))
#        for sample in foldercontent:
#            if sample == '03_500_100H1': continue
#            path=os.path.join(mount,'newReco',folder,sample,'00_raw')
##            if not os.path.exists(path): continue
#            if sample in wetsamples:
#                samples.append([path,sample])
#            #print(sample, path)
#    
#    return samples
	

		
	
		
		
		
		
		
import os
from ij import IJ, ImagePlus, ImageStack

baseFolder = r"D:\Paper_3_wicking_yarn_interlaces\reconstructed_data"

sample="T4_025_3"
slice_number=300

#processing_stage='02_pystack_registered_leg_0'
processing_stage='00_raw'
#processing_stage='03_gradient_filtered'
#processing_stage = '03_reg_ML_seg'


def openSampleImages(folder, imgNumber, name="stack"):
    """Open all images in folder's subfolder corresponding to imgNumber"""
    #folder = "E:\\TOMCAT_02\\pet300-1_seq2_water"
    #folder = "E:\\TOMCAT_02\\pet300-1_seq2"
    #imgNumber = 20;

    names = []
    scans = []
    subfolders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    isFirst = True
    for subfolder in subfolders:
        imgs = os.listdir(folder + "/" + subfolder)
        if len(imgs) >= imgNumber:
            currImp = IJ.openImage(folder + "/" + subfolder + "/" + imgs[imgNumber])
            names.append(imgs[imgNumber])
            scans.append(subfolder)
            if isFirst:
                stack = ImageStack(currImp.getWidth(), currImp.getHeight())
                isFirst = False
            stack.addSlice(currImp.getProcessor())

    imp = ImagePlus("stack", stack)
    imp.show()
    imp.setTitle(name)
    return imp, names, scans


samples = os.listdir(baseFolder)
samples = [sample]

for sample in samples:
	sourceFolder = os.path.join(baseFolder, sample, processing_stage)
	original, names, scans = openSampleImages(sourceFolder, slice_number, name=''.join([sample,"_slice_",str(slice_number)]))
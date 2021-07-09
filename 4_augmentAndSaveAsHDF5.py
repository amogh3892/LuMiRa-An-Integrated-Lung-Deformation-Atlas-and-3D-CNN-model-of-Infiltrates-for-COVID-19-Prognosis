import SimpleITK as sitk 
from pathlib import Path
import sys 
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import tables
import os 
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 
from augmentation3DUtil import Augmentation3DUtil
from augmentation3DUtil import Transforms
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.morphology import opening, closing
from skimage.morphology import disk
from skimage.measure import regionprops
from skimage.measure import label as ConnectedComponent
from skimage.transform import resize as skresize
import pandas as pd 


def _getAugmentedData(imgs,masks,nosamples):
    
    """ 
    This function defines different augmentations/transofrmation sepcified for a single image 
    img,mask : to be provided SimpleITK images 
    nosamples : (int) number of augmented samples to be returned
    
    """
    au = Augmentation3DUtil(imgs,masks=masks)

    au.add(Transforms.SHEAR,probability = 0.2, magnitude = (0.02,0.05))
    au.add(Transforms.SHEAR,probability = 0.2, magnitude = (0.01,0.05))
    au.add(Transforms.SHEAR,probability = 0.2, magnitude = (0.03,0.05))

    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 2)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -2)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 4)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -4)

    au.add(Transforms.FLIPHORIZONTAL,probability = 0.5)

    imgs, augs = au.process(nosamples)

    return imgs,augs

def createHDF5(splitspathname,patchSize,depth):
    
    """
    splitspathname : name of the file (json) which has train test splits info 
    patchSize : x,y dimension of the image 
    depth : z dimension of the image 
    """
    
    outputfolder = fr"outputs/hdf5/{splitspathname}"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    img_dtype = tables.Float32Atom()
    mask_dtype = tables.UInt8Atom()

    if depth > 1:
        shape = (0, depth, patchSize[0], patchSize[1])
        chunk_shape = (1,depth,patchSize[0],patchSize[1])

        sdf_shape = (0,8,depth,patchSize[0],patchSize[1])
        sdf_chunk_shape = (1,8,depth,patchSize[0],patchSize[1])

    else:
        import pdb 
        pdb.set_trace()

    filters = tables.Filters(complevel=5)

    splitspath = fr"outputs/splits/{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)

    phases = np.unique(list(splitsdict.values()))

    for phase in phases:
        hdf5_path = fr'{outputfolder}/{phase}.h5'

        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()

        hdf5_file = tables.open_file(hdf5_path, mode='w')


        data = hdf5_file.create_earray(hdf5_file.root, "data", img_dtype,
                                            shape=shape,
                                            chunkshape = chunk_shape,
                                            filters = filters)

        sdf = hdf5_file.create_earray(hdf5_file.root, "sdf", img_dtype,
                                            shape=sdf_shape,
                                            chunkshape = sdf_chunk_shape,
                                            filters = filters)

        pvalue = hdf5_file.create_earray(hdf5_file.root, "pvalue", img_dtype,
                                            shape=sdf_shape,
                                            chunkshape = sdf_chunk_shape,
                                            filters = filters)

                              
        mask = hdf5_file.create_earray(hdf5_file.root, "mask", mask_dtype,
                                            shape=shape,
                                            chunkshape = chunk_shape,
                                            filters = filters)     

        hdf5_file.close()


def _addToHDF5(imgarr,pvaluearr,maskarr,phase,splitspathname):
    
    """
    imgarr : input image sample 
    pvaluearr: the difference atlas as numpy array
    maskarr : output mask (segmented consolidation regions)
    phase : phase of that image (train,test,val)
    splitspathname : name of the file (json) which has train test splits info 
    """
    outputfolder = fr"outputs/hdf5/{splitspathname}"

    hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')

    data = hdf5_file.root["data"]
    sdf = hdf5_file.root["sdf"]
    pvalue = hdf5_file.root["pvalue"]
    mask = hdf5_file.root["mask"]

    data.append(imgarr[None])
    pvalue.append(pvaluearr[None])
    mask.append(maskarr[None])
    
    hdf5_file.close()

def getAugmentedData(folderpath,name, pvalueimg, nosamples = None):
    
    """
    folderpath : path to folder containing images, mask
    """
    folderpath = Path(folderpath)

    img = sitk.ReadImage(str(folderpath.joinpath(fr"deformRes/result.0.nii.gz")))
    img = DataUtil.resampleimage(img,(2,2,2),img.GetOrigin())

    mask = sitk.ReadImage(str(folderpath.joinpath(fr"manualTrans/result.nii.gz")))
    mask = DataUtil.resampleimage(mask,(2,2,2),img.GetOrigin(),interpolator=sitk.sitkNearestNeighbor)

    ret = []
    
    orgimg,augs = _getAugmentedData([img,pvalueimg],[mask],nosamples)
    ret.append((orgimg))

    if augs is not None:
        for i in range(len(augs)):
            ret.append(augs[i])

    return ret

def normalizeImage(img,_min,_max,clipValue):

    imgarr = sitk.GetArrayFromImage(img)

    if clipValue is not None:
        imgarr[imgarr > clipValue] = clipValue 

    imgarr[imgarr < _min] = _min
    imgarr[imgarr > _max] = _max

    imgarr = (imgarr - _min)/(_max - _min)

    imgarr = imgarr.astype(np.float32)

    return imgarr

def split_cuboid(arr,xsize,ysize,zsize):
    ret = None 
    for i in range(2):
        for j in range(2):
            for k in range(2):
                split = arr[k*zsize:(k+1)*zsize,j*ysize:(j+1)*ysize,i*xsize:(i+1)*xsize]
                ret = split[None] if ret is None else np.vstack((ret,split[None]))

    return ret 


def addToHDF5(img,pvalue,mask,phase,label,name):
    
    """ 
    Collect samples from the cropped volume and add them into HDF5 file 

    img : SimpleITK image to be pre-processed
    pvalue: SimpleITK image of difference atlas 
    mask: Lung mask 
    phase: 'train', 'val' or 'test'.
    label: Label (ventilator or no ventilator)
    name: Case/ Patient name

    """

    # co-ordinates of the lung mask based on the difference atlas 
    startx = 27 
    endx = 171
    starty = 56
    endy = 152
    startz = 4
    endz = 132

    detName = [] 
    detLabel = []

    mask = DataUtil.convert2binary(mask)

    imgarr = sitk.GetArrayFromImage(img)
    pvaluearr = sitk.GetArrayFromImage(pvalue)
    maskarr = sitk.GetArrayFromImage(mask)
    maskarr = maskarr.astype(np.uint8)

    rfarr = rfarr/rfarr.max()

    try:
        img_sample = imgarr[startz:endz, starty:endy, startx:endx]
        pvalue_sample = pvaluearr[startz*2:endz*2, starty*2:endy*2, startx*2:endx*2]
        mask_sample = maskarr[startz:endz, starty:endy, startx:endx]
    except:
        import pdb 
        pdb.set_trace()


    pvalue_sample = split_cuboid(pvalue_sample,144,96,128)

    _addToHDF5(img_sample,pvalue_sample,mask_sample,phase,splitspathname)

    return [name],[label]

if __name__ == "__main__":

    # This script performs augmentations of the dataset and saved all the 
    # images as hdf5 to provide as input to the network 
    # For each of the cross validation fold, train val and test will be saved.

    # labelsdict: Read a labels dictionary with keys being name of the case and
    # values being labels with 0: patient who does not need ventilator
    # and 1: patient who requires a ventilator. 
    labelsdict = DataUtil.readJson("<path to the labels dictionary json file>")

    cvsplits = 3

    for cv in range(cvsplits):
        splitspathname = fr"<path to splits name>"

        # input array size 
        newsize2D = (96,144) 
        depth = 128
        
        splitspath = fr"outputs/splits/{splitspathname}.json"
        splitsdict = DataUtil.readJson(splitspath)

        traincases = [x for x in splitsdict.keys() if splitsdict[x]!='test']
        testcases = [x for x in splitsdict.keys() if splitsdict[x]=='test' and "UH" not in x] 
        uhtestcases = [x for x in splitsdict.keys() if splitsdict[x]=='test' and "UH" in x] 

        trainvalues = [labelsdict[x] for x in traincases]
        testvalues = [labelsdict[x] for x in testcases]
        uhtestvalues = [labelsdict[x] for x in uhtestcases]

        cases = list(splitsdict.keys())

        values = [labelsdict[x] for x in cases]

        createHDF5(splitspathname,newsize2D,depth)

        casenames = {} 
        casenames["train"] = [] 
        casenames["val"] = [] 
        casenames["test"] = [] 

        caselabels = {} 
        caselabels["train"] = [] 
        caselabels["val"] = [] 
        caselabels["test"] = [] 

        for j,name in enumerate(cases):

            if labelsdict[name] == 1:
                cat = "vent"
            else:
                cat = "novent"

            dataset = name.split("_")[0]
            sb = Path(fr"outputs/registered/meantemplate/{name}")

            label = labelsdict[name]
            nosamples = 2 if label == 0 else 2
 
            print(name,j)
            phase = splitsdict[name]

            ret = None 

            if phase == "train":
                ret = getAugmentedData(sb,name,pvalueimg,nosamples=nosamples)
            else:
                ret = getAugmentedData(sb,name,pvalueimg,nosamples=None)

            for k,aug in enumerate(ret):
        
                augimg = aug[0][0]
                augpvalue = aug[0][1]
                augmask = aug[1][0]
                
                _img = augimg
                _sdf = augsdf 
                _pvalue = augpvalue
                _rf = augrf
                _mask = augmask 

                if len(_img.GetSize()) < 3:
                    import pdb 
                    pdb.set_trace()

                casename = name if k == 0 else fr"{name}_A{k}"

                _casenames,_caselabels = addToHDF5(_img,_pvalue,_mask,phase,label,casename)

                casenames[phase].extend(_casenames)
                caselabels[phase].extend(_caselabels)

        outputfolder = fr"outputs/hdf5/{splitspathname}"

        for phase in ["train","test","val"]:
            hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')
            hdf5_file.create_array(hdf5_file.root, fr'names', casenames[phase])
            hdf5_file.create_array(hdf5_file.root, fr'labels', caselabels[phase])
            hdf5_file.close()

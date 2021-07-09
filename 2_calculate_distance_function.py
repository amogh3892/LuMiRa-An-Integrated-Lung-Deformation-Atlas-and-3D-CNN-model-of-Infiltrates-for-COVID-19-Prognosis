import sys
import os  
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import SimpleITK as sitk 
import pandas as pd 
import numpy as np 
import scipy
import skfmm
from skimage import segmentation
from scipy.ndimage.morphology import distance_transform_edt

if __name__ == "__main__":

    # path to the splits dictionary 
    # splits dictionary contains keys as patient/case names; 
    # values as whether belonging to 'train', 'val' or 'test' split
    splitname = "<name of splitsdictionary as json file>"
    splitsdict = DataUtil.readJson(fr"outputs/splits/{splitname}.json")

    names = splitsdict.keys()

    # labelsdict: Read a labels dictionary with keys being name of the case and
    # values being labels with 0: patient who does not need ventilator
    # and 1: patient who requires a ventilator. 
    labelsdict = DataUtil.readJson("<path to the labels dictionary json file>")


    # read template mask and resample it to isotropic volume of spacing (1,1,1)   
    lungmask = sitk.ReadImage(fr"outputs/templatepath/templatemask.nii.gz")
    lungmask = DataUtil.resampleimage(lungmask,(1,1,1),lungmask.GetOrigin(),interpolator=sitk.sitkNearestNeighbor)
    lungmask = sitk.BinaryDilate(lungmask,(5,5,5),sitk.sitkBall)

    for i,name in enumerate(names):
        print(name,i)

        # Read the lung mask 
        inputfolder = fr"outputs/registered/meantemplate/{name}/lungTrans" 
        inputpath = fr"{inputfolder}/result.nii.gz"

        if os.path.exists(inputpath):

            # read lung mask of the image for which distance function has to be calculated
            img = sitk.ReadImage(inputpath)

            # some preprocessing before calculating the distance function 
            img = sitk.BinaryMorphologicalClosing(img,(1,1,0),sitk.sitkBall)
            img = sitk.BinaryMorphologicalClosing(img,(1,1,0),sitk.sitkBall)

            img = DataUtil.resampleimage(img,(1,1,1),img.GetOrigin(),interpolator=sitk.sitkNearestNeighbor)

            # Obtain the boundary points
            insidepoints = sitk.BinaryErode(img,(1,1,0),sitk.sitkBall)
            insidepointsarr = sitk.GetArrayFromImage(insidepoints)

            boundary = sitk.Subtract(sitk.BinaryDilate(img,(1,1,0),sitk.sitkBall),img)
            boundaryarr = sitk.GetArrayFromImage(boundary)
            boundaryarr[boundaryarr == 0] = 2
            boundaryarr[boundaryarr == 1] = 0
            boundaryarr[boundaryarr == 2] = 1 

            # Apply Signed distance function (SDF), based on the bounday points
            # All values inside the boundary negative and outside positive 
            distarr = distance_transform_edt(boundaryarr)
            distarr[insidepointsarr == 1] = distarr[insidepointsarr == 1]*-1

            dist = sitk.GetImageFromArray(distarr)
            dist = DataUtil.copyImageParameters(dist,img)

            # Write SDF 
            sitk.WriteImage(dist,fr"{inputfolder}/{name}-SDF.nii.gz")
            
            dist = sitk.Mask(dist,lungmask)
            sitk.WriteImage(dist,fr"{inputfolder}/{name}-SDF-masked.nii.gz")


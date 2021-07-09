import sys
import os  
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import SimpleITK as sitk 
import shutil 
import pandas as pd 
import numpy as np 

def correctmasks(mask,img):
    arr = sitk.GetArrayFromImage(mask)
    arr[arr < 0] = 0 
    arr[arr > 0] = 1 

    arr = arr.astype(np.uint8)
    ret = sitk.GetImageFromArray(arr)
    ret = DataUtil.copyImageParameters(ret,img)
    return ret

if __name__ == "__main__":
    
    # labelsdict: Read a labels dictionary with keys being name of the case and
    # values being labels with 0: patient who does not need ventilator
    # and 1: patient who requires a ventilator. 
    labelsdict = DataUtil.readJson("<path to the labels dictionary json file>")

    
    names = list(labelsdict.keys())

    # patient to the template image and the template mask (fixedimage for registration)
    fixedpath = fr"<path to template image>"
    fixedmaskpath = fr"<path to template image lung mask>"

    for name in names:
        dataset = name.split("_")[0]

        # foldername of the directory with preprocessed images.
        inputfoldername = "2_Preprocessed"

        inpath = fr"..\Data\{dataset}\{inputfoldername}\{name}"

        print(name)

        # path of the moving image for registration
        movingpath = fr"{inpath}\{name}.nii.gz"

        # paths to lung mask and the segmentation masks of infiltrate regions
        maskpathauto = fr"{inpath}\{name}-ns-label.nii.gz"
        maskpathlung = fr"{inpath}\{name}-lungmask-label.nii.gz"

        # Read fixed and moving images
        img = sitk.ReadImage(movingpath)
        lungmask = sitk.ReadImage(maskpathlung)
        automask = sitk.ReadImage(maskpathauto)

        # correct masks, make sure they are binary. 
        lungmask = correctmasks(lungmask,img)
        automask = correctmasks(automask,img)

        # path to the output folder
        outpath = fr"outputs\registered\meantemplate\{name}"
        DataUtil.mkdir(outpath)

        sitk.WriteImage(img,fr"{outpath}\{name}.nii.gz")
        sitk.WriteImage(lungmask, fr"{outpath}\{name}-lungmask-label.nii.gz")
        sitk.WriteImage(automask, fr"{outpath}\{name}-ns-label.nii.gz")

        movingpath = fr"{outpath}\{name}.nii.gz"

        maskpathauto = fr"{outpath}\{name}-ns-label.nii.gz"
        maskpathlung = fr"{outpath}\{name}-lungmask-label.nii.gz"

        parameterpath = "deformRegistration.txt"
        outputfolder1 = fr"{outpath}\deformRes"
        outputfolder2 = fr"{outpath}\autoTrans"
        outputfolder3 = fr"{outpath}\\lungTrans"
        
        if os.path.exists(outputfolder1):
            shutil.rmtree(outputfolder1)

        if os.path.exists(outputfolder2):
            shutil.rmtree(outputfolder2)

        if os.path.exists(outputfolder3):
            shutil.rmtree(outputfolder3)

        # perform registration, flag the patients with registration error. 
        with open("errorlog.txt","a") as outfile:
            DataUtil.mkdir(outputfolder1)
            DataUtil.mkdir(outputfolder2)
            DataUtil.mkdir(outputfolder3)

            try:
                
                # apply registration and save the transformation parameters
                MedImageProcessingUtil.registerImages(fixedpath,movingpath,parameterpath,outputfolder1,fixedmaskpath=None,movingmaskpath=None)

                # Use the transformation parameters to transform the remaining masks. 
                MedImageProcessingUtil.transformImages(maskpathauto,outputfolder1,outputfolder2,mask=True)
                MedImageProcessingUtil.transformImages(maskpathlung,outputfolder1,outputfolder3,mask=True)

                # pre-process the transformed masks
                deformedimg = sitk.ReadImage(fr"{outputfolder1}\result.0.nii.gz")
                automask = sitk.ReadImage(fr"{outputfolder2}\result.nii.gz")
                lungmask = sitk.ReadImage(fr"{outputfolder3}\result.nii.gz")

                lungmask = correctmasks(lungmask,deformedimg)
                automask = correctmasks(automask,deformedimg)
                manualmask = correctmasks(manualmask,deformedimg)

                sitk.WriteImage(lungmask,fr"{outputfolder3}\result.nii.gz")
                sitk.WriteImage(automask,fr"{outputfolder1}\result.nii.gz")
                sitk.WriteImage(manualmask,fr"{outputfolder2}\result.nii.gz")

            except:
                outfile.write(f"{name}\n")

        outfile.close()

import SimpleITK as sitk 
import numpy as np 
import pandas as pd 
import os 
import cv2
from pathlib import Path 
import subprocess

class MedImageProcessingUtil(object):
    def __init__(self):
        pass 

    @staticmethod
    def copyImageHeaders(toImage,fromImage):
        toImage.SetSpacing(fromImage.GetSpacing())
        toImage.SetDirection(fromImage.GetDirection())
        toImage.SetOrigin(fromImage.GetOrigin())
        return toImage

    @staticmethod
    def cv2inpaint(ipatch,mpatch = None):

        if ipatch.ndim == 2: 
            ipatch = ipatch.astype(np.float32)
        
        elif ipatch.ndim == 3: 
            ipatch = ipatch.astype(np.uint8)
        
        if ipatch[ipatch == 0].size > 0 :
            ipatch = np.pad(ipatch, 1, mode='constant', constant_values=0)

            if mpatch is None:
                mpatch = np.zeros(ipatch.shape)
                mpatch[ipatch == 0] = 1 
                
            else:
                mpatch = np.pad(mpatch, 1, mode='constant', constant_values=0)

            mpatch = mpatch.astype(np.uint8)

            dst = cv2.inpaint(ipatch,mpatch,3,cv2.INPAINT_NS)
            dst = dst[1:-1,1:-1]
        else:
            dst = ipatch
    
        return dst


    @staticmethod
    def registerImages(fixedpath,movingpath,parameterpath,outputfolder,fixedmaskpath=None,movingmaskpath=None):

        cmd = "export PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/bin:$PATH\nexport DYLD_LIBRARY_PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/lib:$DYLD_LIBRARY_PATH\n"

        if (fixedmaskpath and movingmaskpath) is not None:
            cmd = cmd + 'elastix -f {} -fMask {} -m {} -mMask {} -p {} -out {}'.format(fixedpath,fixedmaskpath,movingpath,movingmaskpath,parameterpath,outputfolder)
        elif fixedmaskpath is not None :
            cmd = cmd + 'elastix -f {} -fMask {} -m {} -p {} -out {}'.format(fixedpath,fixedmaskpath,movingpath,parameterpath,outputfolder)
        else:
            cmd = cmd + 'elastix -f {} -m {} -p {} -out {}'.format(fixedpath,movingpath,parameterpath,outputfolder)
    
        if not cmd is None:
            subprocess.call([cmd],shell=True) 
        else:
            print("Masks not provided properly")

    @staticmethod
    def transformImages(movingpath,transformationpath,outputfolder,mask=None):
    
        cmd = "export PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/bin:$PATH\nexport DYLD_LIBRARY_PATH=/Users/amogh3892/Documents/Softwares/elastix/elastix_macosx64_v4/lib:$DYLD_LIBRARY_PATH\n"

        if mask is not None:

            with open(fr"{transformationpath}/TransformParameters.0.txt","r") as infile:
                trans = infile.read()
            infile.close()

            trans = trans.replace('(ResampleInterpolator "FinalLinearInterpolator")','(ResampleInterpolator "FinalNearestNeighborInterpolator")')
            trans = trans.replace('(ResampleInterpolator "FinalBSplineInterpolator")','(ResampleInterpolator "FinalNearestNeighborInterpolator")')
            trans = trans.replace("(FinalBSplineInterpolationOrder 3)","(FinalBSplineInterpolationOrder 0)")

            with open(fr"{transformationpath}/TransformParametersMASK.0.txt","w") as infile:
                infile.writelines(trans)
            infile.close()

            cmd = cmd + "transformix -in {} -tp {}/TransformParametersMASK.0.txt -out {}".format(movingpath,transformationpath,outputfolder)
            subprocess.call([cmd],shell=True) 


        else:
            cmd = cmd + "transformix -in {} -tp {}/TransformParameters.0.txt -out {}".format(movingpath,transformationpath,outputfolder)
            subprocess.call([cmd],shell=True) 






from medImageProcessingUtil import MedImageProcessingUtil
import SimpleITK as sitk 
from glob import glob 
import numpy as np 
import pandas as pd 
import os 
import sys 
import yaml 
from progressbar import *  
import json 
import pickle
from pathlib import Path 
import matplotlib.pyplot as plt 

class DataUtil(object):
    def __init__(self):
        pass 

    @staticmethod
    def createYAML(dct,filepath):
        with open(filepath, 'w') as outfile:
            yaml.dump(dct, outfile, default_flow_style=False)
        outfile.close()

    @staticmethod
    def readYAML(filename):
        with open(filename, 'r') as infile:
            dct  = yaml.load(infile)
        infile.close()

        return dct

    @staticmethod
    def getProgressbar(message,size):
        widgets = [message, Percentage(), ' ', Bar(marker='-',left='[',right=']'),
                ' ', ETA()] #see docs for other options
        pbar = ProgressBar(widgets=widgets, maxval=size)
        return pbar


    @staticmethod
    def convert2binary(img):
        spa = img.GetSpacing()
        ori = img.GetOrigin()
        dire = img.GetDirection()

        imgarr = sitk.GetArrayFromImage(img)
        imgarr[imgarr > 0] = 1
        binimg = sitk.GetImageFromArray(imgarr)

        binimg.SetSpacing(spa)
        binimg.SetOrigin(ori)
        binimg.SetDirection(dire)
        binimg = sitk.Cast(binimg,sitk.sitkUInt8)
        return binimg 


    @staticmethod
    def resampleimage(image, spacing, origin, direction = None, interpolator = sitk.sitkLinear):

        """
        Resamples the image given spacing and the origin with a given interpolator 
        Default interpolator is Linear Interpolator ~ sitk.sitkLinear 

        Other interpolators : sitk.sitkNearestNeighbor , sitk.sitkBSpline,  sitk.sitkGaussian
        
        """
        new_size = tuple([int((image.GetSize()[i]*image.GetSpacing()[i])/spacing[i]) for i in range(len(spacing))])

        if direction is None:
            direction = image.GetDirection()

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(new_size)

        resampled = resampler.Execute(image)

        return resampled 


    @staticmethod
    def resampleimagebysize(image, newsize, interpolator = sitk.sitkLinear):

        """
        Resamples the image to the given size with a given interpolator 
        Default interpolator is Linear Interpolator ~ sitk.sitkLinear 

        Other interpolators : sitk.sitkNearestNeighbor , sitk.sitkBSpline,  sitk.sitkGaussian
        
        """
        size = image.GetSize()
        origin = image.GetOrigin()
        spacing = image.GetSpacing()

        newspacing = [(size[i]/newsize[i])*spacing[i] for i in range(len(size))]

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(newspacing)
        resampler.SetSize(newsize)

        resampled = resampler.Execute(image)

        return resampled 

    @staticmethod
    def renamefilesWithPrefix(datadir,prefix):
        subfolders = glob(fr"{datadir}\**")
        
        for sb in subfolders:
            parentfolder = sb.rsplit("\\",1)[0]
            name = sb.rsplit("\\",1)[-1]
            
            if not prefix in name:

                newname = fr"{prefix}_{name}"
                newsb = fr"{parentfolder}\{newname}"
                os.rename(sb,newsb)

    @staticmethod
    def labelsCSV2Dict(filepath,namecolumn,labelscolumn):
        df = pd.read_csv(filepath)
        labelsdict = dict(zip(df[namecolumn], df[labelscolumn]))
        return labelsdict

    @staticmethod
    def readJson(filepath):
        with open(filepath,'r') as infile:
            dct = json.load(infile)
        infile.close()
        return dct 

    @staticmethod
    def writeJson(dct,filepath):
        with open(filepath,'w') as infile:
            json.dump(dct,infile)
        infile.close()
 

    @staticmethod
    def readPickle(filepath,mode = "rb",encoding = "latin1"):
        with open(filepath,mode) as infile:
            ret = pickle.load(infile,encoding = encoding)
        infile.close()
        return ret 

    @staticmethod
    def writePickle(obj,filepath,mode = "wb",protocol = 2):
        with open(filepath,mode) as infile:
            pickle.dump(obj,infile,protocol=protocol)
        infile.close()


    @staticmethod
    def differenceFiles(path1,path2,ext):

        """
        path1, path2 : full path to files/directories
        ext should be * for directories
        """

        files1 = glob(fr"{path1}\*{ext}")
        files2 = glob(fr"{path2}\*{ext}")

        names1 = [x.split("\\")[-1] for x in files1]
        names2 = [x.split("\\")[-1] for x in files2]

        names1 = set(names1)
        names2 = set(names2)

        return names1.difference(names2), names2.difference(names1)


    @staticmethod
    def GetArrayFromImage(img):
        arr = sitk.GetArrayFromImage(img)
        arr = np.flip(arr,1)
        arr = np.flip(arr,2)
        return arr 


    @staticmethod
    def readDicom(dir):
        """
        A function to convet dicom to sitk image.
        To save the image : sitk.WriteImage(img,"name.format")
        
        """
        img = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dir))

        return img 

    @staticmethod
    def getSubDirectories(dir):
        dirpath = Path(dir)
        subdirs = [x for x in dirpath.iterdir() if x.is_dir()]
        return subdirs


    @staticmethod
    def getFilesWithExtension(parentfolder,ext):

        fileExt = f'*.{ext}'
        filepaths = list(Path(parentfolder).glob(fileExt))

        return filepaths


    @staticmethod
    def mkdir(absolutepath):
        Path(absolutepath).mkdir(parents=True, exist_ok=True)


    @staticmethod
    def copyImageParameters(img,ref):
        img.SetDirection(ref.GetDirection())
        img.SetOrigin(ref.GetOrigin())
        img.SetSpacing(ref.GetSpacing())

        return img 

    @staticmethod
    def biascorrectImage(inputImage,maskpercentile=90,numberFittingLevels=6,iterations=100):
        inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
        # maskImage = sitk.OtsuMultipleThresholds(inputImage)

        arr = sitk.GetArrayFromImage(inputImage)
        threshold = np.percentile(arr,maskpercentile)
        mask = np.zeros(arr.shape)
        mask[arr > threshold] = 1 
        maskImage = sitk.GetImageFromArray(mask)
        maskImage = DataUtil.copyImageParameters(maskImage,inputImage)
        maskImage = DataUtil.convert2binary(maskImage)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([iterations]* numberFittingLevels)

        output = corrector.Execute(inputImage,maskImage)

        return output
import sys 
from dataUtil import DataUtil 
import SimpleITK as sitk 
import numpy as np 
import pandas as pd 
import os 
from glob import glob 

if __name__ == "__main__":

    # This script is mainly to create two commands  
    # merging all signed distance function images to a single file 
    # create a command for randomise function of fsl to obtain the difference altas 
    # see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise/UserGuide for further details.

    # Run mergecmd.bash and getdiffvol.bash once created.


    # labelsdict: Read a labels dictionary with keys being name of the case and
    # values being labels with 0: patient who does not need ventilator
    # and 1: patient who requires a ventilator. 
    labelsdict = DataUtil.readJson("<path to the labels dictionary json file>")


    # path to the splits dictionary 
    # splits dictionary contains keys as patient/case names; 
    # values as whether belonging to 'train', 'val' or 'test' split
    splitname = "<name of splitsdictionary as json file>"

    for cv in range(3):

        # reading the splits dictionary (based on the cross validation fold)
        splitsdict = DataUtil.readJson(fr"outputs/splits/{splitname}_{cv}.json")

        # extract all patient/ case names
        names = splitsdict.keys()
        names = [x for x in names if splitsdict[x]=="train"]

        cnt = 0 

        DataUtil.mkdir(fr"outputs/registered/final/{splitname}_{cv}")

        cmd = fr'fslmerge -t 4D.nii.gz'

        designMat = ""

        for name in names:

            if labelsdict[name] == 1:
                designMat = designMat+"1 0\n"
            else:
                designMat = designMat+"0 1\n"
            
            inputpath = fr"../../meantemplate/{name}/lungTrans/{name}-SDF-masked.nii.gz"

            print(name,labelsdict[name])
            cmd = fr"{cmd} {inputpath}"
            cnt = cnt + 1 

        with open(fr"outputs/registered/final/{splitname}_{cv}/mergecmd.bash","w") as outfile:
            outfile.write(cmd)
        outfile.close()

        designMat = "/NumWaves 2\n/NumPoints {}\n/Matrix\n".format(cnt) + designMat

        with open(fr"outputs/registered/final/{splitname}_{cv}/design.mat","w") as outfile:
            outfile.write(designMat)
        outfile.close()

        designcon = """/NumWaves 2\n/NumPoints 1\n/Matrix\n1 -1"""

        with open(fr"outputs/registered/final/{splitname}_{cv}/design.con","w") as outfile:
            outfile.write(designcon)
        outfile.close()

        lungmask = sitk.ReadImage(fr"outputs/templatepath/templatemask.nii.gz")
        lungmask = DataUtil.resampleimage(lungmask,(2,2,2),lungmask.GetOrigin(),interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(lungmask,fr"outputs/registered/final/{splitname}_{cv}/templatemask.nii")

        randomisecmd = """randomise -i 4D.nii.gz -o diffPValue -d design.mat -t design.con -m templatemask.nii -n 500 -T"""

        with open(fr"outputs/registered/final/{splitname}_{cv}/getdiffvol.bash","w") as outfile:
            outfile.write(randomisecmd)
        outfile.close()


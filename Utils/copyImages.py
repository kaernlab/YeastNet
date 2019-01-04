import pdb
import torch
import shutil
import os
import numpy as np
import argparse



## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_num", type=str, help="Cross Validation model number to use", default="0")
args = parser.parse_args()


def makeImageFolder(model_num, makeYNfolder = False, makeCSfolder = False, makeEPPfolder = True):
    if makeYNfolder:
        ##Copy Images for YeastNet accuracy
        checkpoint = torch.load('../CrossValidation/CrossVal Model Vault/model_cp' + str(model_num) +  '.pt')
        testIDs = checkpoint['testID']

        if not os.path.isdir('../CrossValidation/CrossVal Accuracy/Model' + str(model_num)):
            os.mkdir('../CrossValidation/CrossVal Accuracy/Model' + str(model_num))

        for testID in testIDs:
            filename = '/im%03d.tif' % testID
            orig_path = '../Training Data 1D/Images'
            dest_path = '../CrossValidation/CrossVal Accuracy/model'  + str(model_num)
            shutil.copy2(orig_path + filename, dest_path + filename)


    if makeCSfolder:
        ##Copy Images for CellStar accuracy
        checkpoint = torch.load('../CrossValidation/CrossVal Model Vault/model_cp' + str(model_num) +  '.pt')
        testIDs = checkpoint['testID']

        orig_path = '../../CellStar/Test Images/Cropped/'
        dest_path = '../CrossValidation/CrossVal Accuracy/Model' + str(model_num) + '/CellStar/'

        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)

        for testID in testIDs:
            if testID < 51:
                orig_filename = '/z1/z1_t_000_000_%03d_BF.png' % (testID+1)
                dest_filename = '/z1_t_000_000_%03d_BF.png' % (testID+1)
            elif testID > 101:
                orig_filename = '/z3/z3_t_000_000_%03d_BF.png' % (testID-101)
                dest_filename = '/z3_t_000_000_%03d_BF.png' % (testID-101)
            else:
                orig_filename = '/z2/z2_t_000_000_%03d_BF.png' % (testID-50)
                dest_filename = '/z2_t_000_000_%03d_BF.png' % (testID-50)
            #pdb.set_trace()
            shutil.copy2(orig_path + orig_filename, dest_path + dest_filename)


    if makeEPPfolder:
        orig_path = '../CrossValidation/CrossVal Accuracy/Model' + str(model_num) + '/CellStar/'
        checkpoint = torch.load('../CrossValidation/CrossVal Model Vault/model_cp' + str(model_num) +  '.pt')
        dest_path = '../../EPPackage/CrossValidation/Model' + str(model_num) + '/RawData/'
        testIDs = checkpoint['testID']
        testIDs.sort()

        if not os.path.isdir('../../EPPackage/CrossValidation/Model' + str(model_num)):
            os.mkdir('../../EPPackage/CrossValidation/Model' + str(model_num))
            os.mkdir('../../EPPackage/CrossValidation/Model' + str(model_num) + '/RawData')

        for idx, testID in enumerate(testIDs):
            if testID < 51:
                orig_filename = 'z1_t_000_000_%03d_BF.png' % (testID+1)
            elif testID > 101:
                orig_filename = 'z3_t_000_000_%03d_BF.png' % (testID-101)
            else:
                orig_filename = 'z2_t_000_000_%03d_BF.png' % (testID-50)

            dest_filename = 'BF_frame%03d.png' % (idx)

            shutil.copy2(orig_path + orig_filename, dest_path + dest_filename)

for i in range(1,11):
    makeImageFolder(i) #args.model_num
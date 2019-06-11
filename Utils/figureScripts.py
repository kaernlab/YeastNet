import numpy as np
import torch
import pdb
import imageio
import scipy.io as sio
import pickle
from Utils.helpers import centreCrop

def compareMasks():
    #Load  
    idx = 25
    if idx < 51:
        mask_name = 'z1_t_000_000_%03d_BF_segmentation.mat' % (idx+1)
    elif idx > 101:
        mask_name = 'z3_t_000_000_%03d_BF_segmentation.mat' % (idx-101)
    else:
        mask_name = 'z2_t_000_000_%03d_BF_segmentation.mat' % (idx-50)


    cs_path1 = './CrossValidation/CrossVal Accuracy/Model7/CellStar/segments/z1_t_000_000_025_BF_segmentation.mat'
    cs_path2 = './CrossValidation/CrossVal Accuracy/Model6/CellStar/segments/z2_t_000_000_025_BF_segmentation.mat'
    cs_path3 = './CrossValidation/CrossVal Accuracy/Model10/CellStar/segments/z3_t_000_000_025_BF_segmentation.mat'

    yn_path1 = './CrossValidation/CrossVal Accuracy/Model7/Results/timelapse.pkl' # index 1 is z1_25
    yn_path2 = './CrossValidation/CrossVal Accuracy/Model6/Results/timelapse.pkl' # index 10 is z2_25
    yn_path3 = './CrossValidation/CrossVal Accuracy/Model10/Results/timelapse.pkl' # index 12 is z3_25


    oldMask1 = sio.loadmat('./Images for Figures/t_025_z1_BW.mat')
    oldMask2 = sio.loadmat('./Images for Figures/t_025_z2_BW.mat')
    oldMask3 = sio.loadmat('./Images for Figures/t_025_z3_BW.mat')

    oldMask1 = (oldMask1['BW'] != 0)*1
    oldMask2 = (oldMask2['BW'] != 0)*1
    oldMask3 = (oldMask3['BW'] != 0)*1

    cellStarMask1 = sio.loadmat(cs_path1)
    cellStarMask2 = sio.loadmat(cs_path2)
    cellStarMask3 = sio.loadmat(cs_path3)

    cellStarMask1 = (cellStarMask1['segments'] != 0)*1
    cellStarMask2 = (cellStarMask2['segments'] != 0)*1
    cellStarMask3 = (cellStarMask3['segments'] != 0)*1


    with open(yn_path1, 'rb') as f:
        tl = pickle.load(f)
        yeastNetMask1 = tl.labels[0]
        bwimage1 = tl.imagesBW[0]

    with open(yn_path2, 'rb') as f:
        tl = pickle.load(f)
        yeastNetMask2 = tl.labels[9]
        bwimage2 = tl.imagesBW[9]

    with open(yn_path3, 'rb') as f:
        tl = pickle.load(f)
        yeastNetMask3 = tl.labels[11]
        bwimage3 = tl.imagesBW[11]

    yeastNetMask1 = (yeastNetMask1 != 0)*1
    yeastNetMask2 = (yeastNetMask2 != 0)*1
    yeastNetMask3 = (yeastNetMask3 != 0)*1


    yeastNetMask1 = centreCrop(yeastNetMask1, 512)
    yeastNetMask2 = centreCrop(yeastNetMask2, 512)
    yeastNetMask3 = centreCrop(yeastNetMask3, 512)

    oldMask1 = centreCrop(oldMask1, 512)
    oldMask2 = centreCrop(oldMask2, 512)
    oldMask3 = centreCrop(oldMask3, 512)

    cellStarMask1 = centreCrop(cellStarMask1, 512)
    cellStarMask2 = centreCrop(cellStarMask2, 512)
    cellStarMask3 = centreCrop(cellStarMask3, 512)

    bwimage1 = centreCrop(bwimage1, 512)
    bwimage2 = centreCrop(bwimage2, 512)
    bwimage3 = centreCrop(bwimage3, 512)


    imageio.imwrite('yeastNetMask1.png', yeastNetMask1)
    imageio.imwrite('yeastNetMask2.png', yeastNetMask2)
    imageio.imwrite('yeastNetMask3.png', yeastNetMask3)

    imageio.imwrite('oldMask1.png', oldMask1)
    imageio.imwrite('oldMask2.png', oldMask2)
    imageio.imwrite('oldMask3.png', oldMask3)

    imageio.imwrite('cellStarMask1.png', cellStarMask1)
    imageio.imwrite('cellStarMask2.png', cellStarMask2)
    imageio.imwrite('cellStarMask3.png', cellStarMask3)
    
    imageio.imwrite('bwimage1.png', bwimage1)
    imageio.imwrite('bwimage2.png', bwimage2)
    imageio.imwrite('bwimage3.png', bwimage3)
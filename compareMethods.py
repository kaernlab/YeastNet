import imageio as imio
import numpy as np
import os
import pdb
from validateNetwork import validate 
import torch
from torch import optim
import scipy.io as sio

## Import Custom Code
from YeastSegmentationDataset import YeastSegmentationDataset
from defineNetwork import Net
from WeightedCrossEntropyLoss import WeightedCrossEntropyLoss
from Utils.helpers import accuracy



def compareOld():
    runningIntOfUnion = 0
    runningPixAccuracy = 0
    for idx in range(51):

    ## Load true labels 
        predmask = sio.loadmat('Test Accuracy/Old Method/t_%.3d.mat' % (idx+1))
        predmask = (predmask['LAB_orig'] != 0)*1
        truemask = sio.loadmat('Test Accuracy/True/mask' + str(idx) + '.mat')
        truemask = (truemask['LAB_orig'] != 0)*1

    ## get accuracy
        IntOfUnion, pixelAccuracy = accuracy(truemask, predmask)
        runningIntOfUnion += IntOfUnion[1]
        runningPixAccuracy += pixelAccuracy

    IntOfUnionOldMethod = runningIntOfUnion / 51


    ## Instantiate Net, Load Parameters, Move Net to GPU
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load("Current Model/model_cp.pt")
    testIDs = checkpoint['testID']
    net.load_state_dict(checkpoint['network'])

    testDataSet = YeastSegmentationDataset(testIDs)
    testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=1,
                                            shuffle=False, num_workers=0)

    ## Set Training hyperparameters/conditions
    criterion = WeightedCrossEntropyLoss()
    IntOfUnionNewMethod = validate(net, device, testLoader, criterion, saveImages = False)

    print(IntOfUnionNewMethod, IntOfUnionOldMethod)

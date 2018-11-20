import imageio as imio
import numpy as np
import os
import pdb
from validateNetwork import validate 
import torch
from torch import optim
import scipy.io as sio
from processImages import YeastSegmentationDataset
from defineNetwork import Net
from weightedLoss import WeightedCrossEntropyLoss
import PIL
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import inferNetwork

def accuracy(true_mask, pred_mask):
    IntOfUnion = np.zeros(2)
    true_bg = (true_mask==0)*1
    true_cl = true_mask
    pred_bg = (pred_mask==0)*1
    pred_cl = pred_mask
    ## Calculate IOU
    Union = np.logical_or(true_bg, pred_bg)
    Intersection = np.logical_and(true_bg, pred_bg)
    IntOfUnion[0] = np.sum(Intersection) / np.sum(Union)
    Union = np.logical_or(true_cl, pred_cl)
    Intersection = np.logical_and(true_cl, pred_cl)
    IntOfUnion[1] = np.sum(Intersection) / np.sum(Union)
    PixAccuracy = true_mask[true_mask==pred_mask].size / true_mask.size

    return IntOfUnion[1]

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
        IntOfUnion = accuracy(truemask, predmask)
        runningIntOfUnion += IntOfUnion

    IntOfUnionOld = runningIntOfUnion / 51


    ## Instantiate Net, Load Parameters, Move Net to GPU
    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load("Current Model/model_cp.pt")
    testIDs = checkpoint['testID']
    trainIDs = checkpoint['trainID']
    iteration = checkpoint['iteration']
    start = checkpoint['epoch']
    net.load_state_dict(checkpoint['network'])

    testDataSet = YeastSegmentationDataset(testIDs)
    testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=1,
                                            shuffle=False, num_workers=0)

    ## Set Training hyperparameters/conditions
    criterion = WeightedCrossEntropyLoss()
    classes = ('background','cell')
    IntOfUnionNew = validate(net, device, testLoader, criterion, saveImages = False)

    print(IntOfUnionNew, IntOfUnionOld)

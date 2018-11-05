## Import Libraries and Modules
import torch
from torch.utils.data import DataLoader
import imageio
import numpy as np

#import pdb
import pdb

## Import Custom Code
from weightedLoss import WeightedCrossEntropyLoss


def validate(net, device, testLoader, criterion, saveImages = False):

    ## Run net without regularization techniques
    net.eval()

    ## Loss Sum accumulator for output
    runningIOU = 0

    ## Loop over batches
    for i, data in enumerate(testLoader, 0):
        ## Get inputs and transfer to GPU
        validationImage, mask, lossWeightMap = data
        validationImage, mask, lossWeightMap = validationImage.to(device), mask.to(device), lossWeightMap.to(device)
       
        ## Run Batch through data with no grad
        with torch.no_grad():
            outputs = net(validationImage.float())
        predictions = outputs.cpu().detach().numpy()[0,:,:,:]
        maskPrediction = (predictions[1] > predictions[0]) * 1
        ## Calculate the Loss for the batch
        #loss = criterion(outputs, mask.long(), lossWeightMap)
        
        PixAccuracy, IntOfUnion = accuracy(mask.cpu().detach().numpy()[0,:,:,0], maskPrediction)
        runningIOU += IntOfUnion[1]
        #print(loss.item())
        ## Output Images
        if saveImages:
            
            imageio.imwrite('Validation/' + str(i) + 'Pred.png', maskPrediction.astype(float))
            imageio.imwrite('Validation/' + str(i) + 'IMG.png', validationImage[0,0,:,:].cpu().detach().numpy())
            imageio.imwrite('Validation/' + str(i) + 'True.png', mask[0,:,:,0].cpu().detach().numpy())

    ## Return the mean Loss
    return runningIOU / i


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

    return PixAccuracy, IntOfUnion








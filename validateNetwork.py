## Import Libraries and Modules
import torch
from torch.utils.data import DataLoader
import imageio
import numpy

#import pdb

## Import Custom Code
from weightedLoss import WeightedCrossEntropyLoss


def validate(net, device, testLoader, criterion, saveImages = False):

    ## Run net without regularization techniques
    net.eval()

    ## Loss Sum accumulator for output
    runningLoss = 0

    ## Loop over batches
    for i, data in enumerate(testLoader, 0):
        ## Get inputs and transfer to GPU
        validationImage, mask, lossWeightMap = data
        validationImage, mask, lossWeightMap = validationImage.to(device), mask.to(device), lossWeightMap.to(device)
       
        ## Run Batch through data with no grad
        with torch.no_grad():
            outputs = net(validationImage.float())

        ## Calculate the Loss for the batch
        loss = criterion(outputs, mask.long(), lossWeightMap)
        runningLoss += loss.item()
        #print(loss.item())
        ## Output Images
        if saveImages:
            bg = outputs.cpu().detach().numpy()[0,0,:,:]
            cl = outputs.cpu().detach().numpy()[0,1,:,:]
            mk = numpy.zeros((512,512))
            mk = (cl>bg)*1
            imageio.imwrite('Validation/' + str(i) + 'Pred.png', mk.astype(float))
            imageio.imwrite('Validation/' + str(i) + 'IMG.png', validationImage[0,0,:,:].cpu().detach().numpy())
            imageio.imwrite('Validation/' + str(i) + 'True.png', mask[0,:,:,0].cpu().detach().numpy())

    ## Return the mean Loss
    return runningLoss / i








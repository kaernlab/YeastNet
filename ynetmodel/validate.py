## Import Libraries and Modules
import torch
import imageio
import numpy as np
import pdb

## Import Custom Code
from Utils.helpers import accuracy

def infer(net, device, testLoader, criterion = None, saveImages = False):

    ## Run net without regularization techniques
    net.eval()

    ## Loss Sum accumulator for output
    runningIOU = 0

    ## Loop over batches
    for i, data in enumerate(testLoader, 1):
        ## Get inputs and transfer to GPU
        image, mask, _ = data
        image = image.to(device)
       
        ## Run Batch through data with no grad
        with torch.no_grad():
            outputs = net(image.float())
        predictions = outputs.cpu().detach().numpy()[0,:,:,:]
        maskPrediction = (predictions[1] > predictions[0]) * 1

        ## Calculate Accuracy and update running total
        _, IntOfUnion = accuracy(mask.cpu().detach().numpy()[0,:,:,0], maskPrediction)
        runningIOU += IntOfUnion[1]
        
        ## Output Images
        if saveImages:
            #pdb.set_trace()
            imageio.imwrite('Validation/' + str(i) + 'Pred.png', (maskPrediction.astype('uint8'))* 255)
            imageio.imwrite('Validation/' + str(i) + 'IMG.png', (image[0,0,:,:].cpu().detach().numpy() * 255).astype('uint8'))
            imageio.imwrite('Validation/' + str(i) + 'True.png', (mask[0,:,:,0].cpu().detach().numpy() * 255).astype('uint8'))

    ## Return the mean Loss
    return runningIOU / i











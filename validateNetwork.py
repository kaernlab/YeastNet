## Libraries and Modules
import torch
from torch.utils.data import DataLoader
from torch import optim
import imageio
import numpy
#import pdb

## Custom Code
import processImages as pi
from processImages import YeastSegmentationDataset
from defineNetwork import Net
from weightedLoss import WeightedCrossEntropyLoss


def validate(net, device, dataset):
    ## Load validation data
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=0)

    criterion = WeightedCrossEntropyLoss()
    net.eval()
    running_loss = 0
    

    ## Loop over batches
    for i, data in enumerate(testloader, 0):
        ## Get inputs and transfer to GPU
        training_image, labels, loss_weight_map = data
        training_image, labels, loss_weight_map = training_image.to(device), labels.to(device), loss_weight_map.to(device)
        
        ## Run Batch through data with no grad
        with torch.no_grad():
            outputs = net(training_image.float())
        ## Calculate the Loss for the batch
        loss = criterion(outputs, labels.long(), loss_weight_map)
        running_loss += loss.item()
        ## Output Images
        #print(1)
        #bg = outputs.cpu().detach().numpy()[0,0,:,:]
        #cl = outputs.cpu().detach().numpy()[0,1,:,:]
        #mk = numpy.zeros((512,512))
        #mk = (cl>bg)*1
        #imageio.imwrite('Inference/' + str(i) + 'Pred.png', mk.astype(float))
        #imageio.imwrite('Inference/' + str(i) + 'IMG.png', training_image[0,0,:,:].cpu().detach().numpy())
        #imageio.imwrite('Inference/' + str(i) + 'True.png', labels[0,:,:,0].cpu().detach().numpy())
        #print(2)
    ## Return the mean Loss
    return running_loss / i








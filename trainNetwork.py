## Import Librarys and Modules
import torch
import pdb
import random
import time
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

## Import Custom Modules
import ynetmodel.validateNetwork as validateNetwork
from Utils.helpers import getDatasetMoments

## Import Custom Classes
from ynetmodel.YeastSegmentationDataset import YeastSegmentationDataset
from ynetmodel.defineNetwork import Net
from ynetmodel.WeightedCrossEntropyLoss import WeightedCrossEntropyLoss


def main():
## Set K-fold, checking if its an environment variable
    try:
        k = int(os.environ['K_FOLD'])
        toResume = os.environ['RESUME']
        normtype = os.environ['NORMTYPE']
        allDatasets = os.environ['ALLDATASETS']
    except KeyError:
        k = 5
        toResume = 'True'
        normtype = 3
        allDatasets = 'True'

    print('Training K-fold ' + str(k) + ': Normalization type ' + str(normtype))


    ## Start Timer, Tensorboard
    start_time = time.time()
    end = 10000
    writer = SummaryWriter(comment='K_Fold' + str(k))#log_dir="./logs")

    ## Instantiate Net, Load Parameters, Move Net to GPU
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)


    ##Send Model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)


    if toResume=='True':
        ## Load State
        if allDatasets == 'False':
            inputpath ="./NewNormModels/new_norm_testDSV%01dK%01d.pt" % (normtype,k) 
        else:
            inputpath ="./NewNormModels/new_norm_testV%01dK%01d.pt" % (normtype,k) 
       
        checkpoint = torch.load(inputpath, map_location=device)
        testIDs = checkpoint['testID']
        trainIDs = checkpoint['trainID']
        iteration = checkpoint['iteration']
        start = checkpoint['epoch']
        highestAccuracy = checkpoint['highestAccuracy']
        net.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        setMoments = getDatasetMoments(trainIDs)
    else:
        ## Intialize State
        iteration = 0
        start = 0
        highestAccuracy = 0
        testIDs = torch.load("./CrossValidation/testIDs.pt")[k-1]
        trainIDs = torch.load("./CrossValidation/trainIDs.pt")[k-1]

        if allDatasets == 'False':
            testIDs.pop('YITDataset1', None)
            testIDs.pop('YITDataset3', None)
            trainIDs.pop('YITDataset1', None)
            trainIDs.pop('YITDataset3', None)

        setMoments = getDatasetMoments(trainIDs)

    ## Parallelize net
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    ##Change Optimizer params
    for g in optimizer.param_groups:
        g['lr'] = 0.05
        #g['momentum'] = 0.9

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=25, verbose=True)


    ## Instantiate Training and Validation DataLoaders
    trainDataSet = YeastSegmentationDataset(trainIDs, crop_size = 256, random_rotate = True, random_flip = False,
                                            no_og_data = False, random_crop=False, normtype=normtype, setMoments = setMoments)
    trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=1,
                                            shuffle=True, num_workers=1)

    testDataSet = YeastSegmentationDataset(testIDs, normtype=normtype, setMoments = setMoments)
    testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=1,
                                            shuffle=False, num_workers=1)


    ## Set Training hyperparameters/conditions
    criterion = WeightedCrossEntropyLoss()
    classes = ('background','cell')

    ## Epoch Loop: first loops over batches, then over v alidation set
    for epoch in range(start,end):  
        
        net.train()
        ## Batch Loop
        for i, data in enumerate(trainLoader, 0):
            ## Total iteration
            iteration+=1

            ## Get inputs
            trainingImage, mask, lossWeightMap = data
            trainingImage, mask, lossWeightMap = trainingImage.to(device), mask.to(device), lossWeightMap.to(device)

            ## Zero the parameter gradients
            optimizer.zero_grad()

            ## Forward Pass
            outputs = net(trainingImage.float())

            ## Write Graph
            #writer.add_graph(net, trainingImage.float())

            ## Calculate and Write Loss
            if torch.isnan(outputs).any() > 0:
                print('There Be Nans')
            loss = criterion(outputs, mask.long(), lossWeightMap)
            
            ## Backpropagate Loss
            loss.backward()

            ## Update Parameters
            
            optimizer.step()

        ## Epoch validation

        val_acc = validateNetwork.validate(net, device, testLoader, criterion, saveImages = True)
        scheduler.step(val_acc)
        #print('Current LR is: ' + str(optimizer.param_groups['lr']))
        
        print('[%d, %d] IntOfUnion (Cell): %.5f \n' % (iteration, epoch + 1, val_acc))
        writer.add_scalar('Validation Cell IOU', val_acc, global_step=epoch, walltime=time.time())
        ## Epoch Time
        elapsed_time = time.time() - start_time
        print(str(elapsed_time / 60) + 'min')

        if val_acc > highestAccuracy:
            save_option = True
            highestAccuracy = val_acc
        else:
            save_option = False

        ## Save Model 
        if save_option: #saveCP:
            try:
                net_state_dict = net.module.state_dict()
            except AttributeError:
                net_state_dict = net.state_dict()

            checkpoint = {
                "network": net_state_dict,
                "optimizer": optimizer.state_dict(),
                "trainID": trainIDs,
                "testID": testIDs,
                "epoch": epoch,
                "iteration": iteration,
                "highestAccuracy": val_acc,
            }
            if allDatasets == 'False':
                outputpath ="./NewNormModels/new_norm_testDSV%01dK%01d.pt" % (normtype,k) 
            else:
                outputpath ="./NewNormModels/new_norm_testV%01dK%01d.pt" % (normtype,k) 
            torch.save(checkpoint, outputpath)

    ## Finish
    elapsed_time = time.time() - start_time
    print('Finished Training, Duration: seconds' + str(elapsed_time))
    writer.close()


if __name__ == '__main__':
    main()
## Import Librarys and Modules
import torch
import pdb
import yaml
import random
import time
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

## Import Custom Modules
from ynetmodel.detect import validate
from Utils.helpers import getDatasetMoments
from ynetmodel.defineNetwork import Net
from ynetmodel.WeightedCrossEntropyLoss import WeightedCrossEntropyLoss
from ynetmodel.YeastSegmentationDataset import YeastSegmentationDataset

def main():

    ## Print Name/Info about Model Training Session
    print('------------------------------------------------------')
    print('|  Training with w0={}, sigma={}.\n|  Training Datasets: {}.\n|  Validation Datasets: {}'.format(loss_param[0], loss_param[1], trainingSets, testingSets),flush=True)
    print('------------------------------------------------------')
    ## Start Timer, Tensorboard
    start_time = time.time()
    writer = SummaryWriter(comment='_12test3,w0={},sigma={}'.format(loss_param[0], loss_param[1]))#log_dir="./logs")

    ## Instantiate Net/Optimizer
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

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

    else:
        ## Intialize State
        iteration = 0
        start = 0
        highestAccuracy = 0 

        ## New two Dataset Training
        #if trainTwo == 'True':
        #    trainIDs = {
        #        'DSDataset': list(range(150)),
        #        'YITDataset1': list(range(60)),
        #        #'YITDataset3': list(range(20)),
        #    }
        #    testIDs = {
        #        #'DSDataset': list(range(150)),
        #        #'YITDataset1': list(range(60)),
        #        'YITDataset3': list(range(20)),
        #    }
        #trainIDs = {dataset: torch.load('./Utils/TrainingSplits/trainIDs.pt')[dataset] for dataset in trainingSets}
        #testIDs = {dataset: torch.load('./Utils/TrainingSplits/testIDs.pt')[dataset] for dataset in testingSets}
        trainIDs = {'DSDataset': list(range(150))}
        testIDs = {'DSDataset': list(range(150))}


    ## Get Statistics of datasets
    trainSetMoments = getDatasetMoments(trainIDs)
    testSetMoments = getDatasetMoments(testIDs)

    ## Parallelize net
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=50, verbose=True)

    ## Instantiate Training and Validation DataLoaders
    trainDataSet = YeastSegmentationDataset(trainIDs, crop_size = crop_size, random_rotate = random_rotate, random_flip = random_flip,
                                            no_og_data = no_og_data, random_crop=random_crop,
                                            setMoments = trainSetMoments, loss_param=loss_param)
    trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=1,
                                            shuffle=True, num_workers=0)
                                            
    testDataSet = YeastSegmentationDataset(testIDs, setMoments = testSetMoments)
    testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=1,
                                            shuffle=False, num_workers=0)

    ## Set Training hyperparameters/conditions
    criterion = WeightedCrossEntropyLoss()
    classes = ('background','cell')

    ## Epoch Loop: first loops over batches, then over validation set
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

            ## Calculate and Write Loss
            if torch.isnan(outputs).any() > 0:
                print('There Be Nans')
            loss = criterion(outputs, mask.long(), lossWeightMap)
            
            ## Backpropagate Loss
            loss.backward()

            ## Update Parameters
            
            optimizer.step()

        ## Epoch validation

        val_acc = validate(net, device, testLoader, criterion, saveImages = True)
        scheduler.step(val_acc)
        
        print('[%d, %d] IntOfUnion (Cell): %.5f \n' % (iteration, epoch + 1, val_acc),flush=True)
        writer.add_scalar('Validation Cell IOU', val_acc, global_step=epoch, walltime=time.time())
        ## Epoch Time
        elapsed_time = time.time() - start_time
        print(str(elapsed_time / 60) + 'min',flush=True)

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
            
            #outputpath = modelfolder + '{}_w0={}_sigma={}.pt'.format(trainingSets, loss_param[0], loss_param[1])
            outputpath = modelfolder + modelname
            torch.save(checkpoint, outputpath)

    ## Finish
    elapsed_time = time.time() - start_time
    print('Finished Training, Duration: seconds' + str(elapsed_time),flush=True)
    with open('ScreenOutput.txt','a') as outputfile:
        outputfile.writelines('w0={},sigma={},accuracy={:.4f}\n'.format(loss_param[0], loss_param[1], 
                                                                checkpoint['highestAccuracy']))
    writer.close()

if __name__ == '__main__':

## Load Settings YAML file
    if os.path.exists("settings.yml"):        
        settings=yaml.load(open("settings.yml","r"))
    train_param = {}
## Try loading settings from environment variable
    try:
        k = int(os.environ['K_FOLD'])
        toResume = os.environ['RESUME']
        normtype = os.environ['NORMTYPE']
        #allDatasets = os.environ['ALLDATASETS']
        loss_param = [os.environ['W0'],os.environ['SIGMA']]
        dataset = os.environ['DATASET']
    except KeyError:
        k = settings['train_param']['k']
        toResume = settings['train_param']['toResume']
        normtype = settings['train_param']['normtype']
        #allDatasets = settings['data']['allDatasets']
        loss_param = settings['data']['loss_param']
        trainingSets = settings['data']['trainingSets']
        testingSets = settings['data']['testingSets']
        

    #loss_param = ['10','5'] if dataset == 'DSDataset' else ['5','5']

    end = settings['train_param']['end']
    lr = settings['train_param']['learning_rate']
    momentum = settings['train_param']['momentum']
    modelfolder = settings['model']['folderpath']
    no_og_data = settings['train_param']['no_og_data']
    random_flip = settings['train_param']['random_flip']
    random_rotate = settings['train_param']['random_rotate']
    random_crop = settings['train_param']['random_crop']
    crop_size = settings['train_param']['crop_size']
    batch_size = settings['train_param']['batch_size']

    main()
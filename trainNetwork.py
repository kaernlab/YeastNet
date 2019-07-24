## Import Librarys and Modules
import torch
import tensorboardX as tbX
import pdb
import random
import time
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch import optim

## Import Custom Modules
import ynetmodel.validateNetwork as validateNetwork

## Import Custom Classes
from ynetmodel.YeastSegmentationDataset import YeastSegmentationDataset
from ynetmodel.defineNetwork import Net
from ynetmodel.WeightedCrossEntropyLoss import WeightedCrossEntropyLoss

## Start Timer, Tensorboard
start_time = time.time()
writer = tbX.SummaryWriter()#log_dir="./logs")
k = 8
end = 10000

## Instantiate Net, Load Parameters, Move Net to GPU
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)


##Send Model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

## Load State
highestAccuracy = 0 #
#checkpoint = torch.load("./CrossValidation/WarmStartModels/model_cp%01d.pt" % k)
checkpoint = torch.load("./CrossValidation/DoubleTrainedModels/model_cp%01d.pt" % k)
#checkpoint = torch.load("./testModel%01d.pt" % k)
testIDs = checkpoint['testID']
trainIDs = checkpoint['trainID']
iteration = checkpoint['iteration']
start = checkpoint['epoch']
#highestAccuracy = checkpoint['highestAccuracy']
net.load_state_dict(checkpoint['network'])
optimizer.load_state_dict(checkpoint['optimizer'])

## Parallelize net
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

##Change Optimizer params
for g in optimizer.param_groups:
    g['lr'] = 0.01
    g['momentum'] = 0.9



## Instantiate Training and Validation DataLoaders
trainDataSet = YeastSegmentationDataset(trainIDs, crop_size = 256, random_rotate = True, random_flip = True,
                                        no_og_data = False, random_crop=True)
trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=10,
                                        shuffle=True, num_workers=10)

testDataSet = YeastSegmentationDataset(testIDs)
testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=1,
                                        shuffle=False, num_workers=10)

#for i in range(len(testDataSet)):
#    data = testDataSet[i]
#    trainingImage, mask, lossWeightMap = data
#pdb.set_trace()

## Set Training hyperparameters/conditions
criterion = WeightedCrossEntropyLoss()
classes = ('background','cell')

## Epoch Loop: first loops over batches, then over v alidation set
for epoch in range(start,end):  
    
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
        #print('Forward Pass')

        ## Write Graph
        writer.add_graph(net, trainingImage.float())

        ## Calculate and Write Loss
        loss = criterion(outputs, mask.long(), lossWeightMap)
        #print('Loss Calculated:', loss.item())
        writer.add_scalar('Batch Loss', loss.item(), global_step=iteration, walltime=time.time())
        
        ## Backpropagate Loss
        loss.backward()
        #print('Backpropagation Done')

        ## Update Parameters
        optimizer.step()
        #print('optimizer')


    ## Epoch validation
    #print('\n\nValidating.... Please Hold')



    val_acc = validateNetwork.validate(net, device, testLoader, criterion)
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
        torch.save(checkpoint, "./CrossValidation/DoubleTrainedModels/model_cp%01d.pt" % k)

## Finish
elapsed_time = time.time() - start_time
print('Finished Training, Duration: seconds' + str(elapsed_time))
writer.close()

# What works Best: make empty Net, load model, make DP model, save using module.state_dict() 
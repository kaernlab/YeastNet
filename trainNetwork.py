## Import Librarys and Modules
import torch
from torch.utils.data import DataLoader
from torch import optim
import tensorboardX as tbX
import pdb
import random
import time
import numpy

## Import Custom Modules
import processImages as pi
import validateNetwork as valNet

## Import Custom Classes
from processImages import YeastSegmentationDataset
from defineNetwork import Net
from weightedLoss import WeightedCrossEntropyLoss

## Start Timer
start_time = time.time()

## Launch Tensorboard Summary Writing Object
writer = tbX.SummaryWriter()#log_dir="./logs")



iteration = 0
start = 0
resume = True
if resume:
    start = 1803
    iteration = 30634
    batchIDs = numpy.load('batchIDs.npy').item()
else:
    ## Make Test and Validation Partitions
    samplingList = list(range(pi.num_images()))
    samples = random.sample(samplingList,153)
    trainingIDs = samples[:129]
    testIDs = samples[130:]
    batchIDs = {
        "train": trainingIDs,
        "test": testIDs
    } 
    numpy.save('batchIDs.npy', batchIDs) #Need to do .item() when loading


## Instantiate Net, Load Parameters, Move Net to GPU
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)#, momentum=0.9)

##Send Model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

## Load State
#net.load_state_dict(torch.load("model.pt"))
#optimizer.load_state_dict(torch.load("model_opt.pt"))

## Instantiate Training and Validation DataLoaders
trainDataSet = YeastSegmentationDataset(batchIDs['train'], crop_size = 256)
trainLoader = torch.utils.data.DataLoader(trainDataSet, batch_size=8,
                                          shuffle=True, num_workers=0)

testDataSet = YeastSegmentationDataset(batchIDs['test'], crop_size = 512)
testLoader = torch.utils.data.DataLoader(testDataSet, batch_size=1,
                                         shuffle=False, num_workers=0)


## Set Training hyperparameters/conditions
#optimizer = optim.SGD(net.parameters(), lr=0.1)#, momentum=0.9, weight_decay=0.0005)
criterion = WeightedCrossEntropyLoss()#nn.CrossEntropyLoss()#
classes = ('background','cell')



# Epoch Loop: first loops over batches, then over v alidation set
for epoch in range(start,10000):  
    
    ## Batch Loop
    for i, data in enumerate(trainLoader, 0):
        ## Total iteration
        iteration+=1

        ## Get inputs
        trainingImage, mask, lossWeightMap = data
        trainingImage, mask, lossWeightMap = trainingImage.to(device), mask.to(device), lossWeightMap.to(device)
        pdb.set_trace()
        ## Zero the parameter gradients
        optimizer.zero_grad()

        ## Forward Pass
        outputs = net(trainingImage.float())
        print('Forward Pass')

        ## Write Graph
        writer.add_graph(net, trainingImage.float())

        ## Calculate and Write Loss
        loss = criterion(outputs, mask.long(), lossWeightMap)
        print('Loss Calculated:', loss.item())
        writer.add_scalar('Batch Loss', loss.item(), iteration)
        
        ## Backpropagate Loss
        loss.backward()
        print('Backpropagation Done')

        #for param in net.parameters():
        #    print(param.grad.data.sum())
        
        #pdb.set_trace()
        ## Update Parameters
        optimizer.step()
        print('optimizer')


    ## Epoch validation
    print('Validating.... Please Hold')
    val_loss = valNet.validate(net, device, testLoader, criterion, saveImages=True)
    print('[%d, %d] loss: %.5f' % (iteration, epoch + 1, val_loss))
    writer.add_scalar('Validation Loss', val_loss, epoch)
    ## Save Model 
    #if True #saveCP:
    torch.save(net.state_dict(),  "model.pt")
    torch.save(optimizer.state_dict(),  "model_opt.pt")
    #numpy.save('batchIDs.npy', resume)

## Finish
elapsed_time = time.time() - start_time
print('Finished Training, Duration: seconds' + str(elapsed_time))#format(elapsed_time, '02d')))
writer.close()
torch.save(net.state_dict(),  "model.pt")

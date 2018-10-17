import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy
import imageio
import tensorboardX as tbX
import pdb
import processImages as pi

from processImages import YeastSegmentationDataset
from defineNetwork import Net
from weightedLoss import WeightedCrossEntropyLoss

yeast_dataset = YeastSegmentationDataset()

writer = tbX.SummaryWriter()#log_dir="./logs")
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.000001)#, momentum=0.9, weight_decay=0.0005)

criterion = WeightedCrossEntropyLoss()#nn.CrossEntropyLoss()#

trainloader = torch.utils.data.DataLoader(yeast_dataset, batch_size=1,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(yeast_dataset, batch_size=1,
                                         shuffle=False, num_workers=0)

classes = ('background','cell')



for epoch in range(10):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Total iteration
        iteration = (1 + i + (epoch)*51)
        best_model_loss = 10

        # Get inputs
        inputs, labels, loss_weight_map = data
        inputs, labels, loss_weight_map = inputs.to(device), labels.to(device), loss_weight_map.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # Forward Pass
        outputs = net(inputs.float())
        print('Forward Pass')

        # Write Graph
        writer.add_graph(net, inputs.float())

        # Output Image
        #print(outputs.detach().numpy().shape)
        bg = outputs.cpu().detach().numpy()[0,0,:,:]
        cl = outputs.cpu().detach().numpy()[0,1,:,:]
        #mk = numpy.zeros((512,512))
        mk = (cl>bg)*1
        print(mk)
        imageio.imwrite('outputs/' + str(iteration) + 'Pred.png', mk.astype(float))
        imageio.imwrite('outputs/' + str(iteration) + 'True.png', pi.convert_image(inputs[0].cpu().detach().numpy()))
        
        # Calculate Loss
        loss = criterion(outputs, labels.long(), loss_weight_map)
        print('Loss Calculated')
        writer.add_scalar('loss', loss.item(), iteration)

        # Backpropagate Loss
        loss.backward()
        print('Backpropagation Done')

        # Update Parameters
        optimizer.step()
        print('optimezer')

        # Print statistics
        print('[%d, %d, %5d] loss: %.5f' % (iteration, epoch + 1, i + 1, loss.item()))

        # Save Model 
        #if best_model_loss>loss.item():
            #torch.save(net.state_dict(),  "model.pt")

print('Finished Training')
#pdb.set_trace()
#writer.export_scalars_to_json("./logs/all_scalars.json")
writer.close()


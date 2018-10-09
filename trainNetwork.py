import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy

from processImages import YeastSegmentationDataset
from defineNetwork import Net
yeast_dataset = YeastSegmentationDataset()

net = Net()

optimizer = optim.SGD(net.parameters(),
                        lr=0.1,
                        momentum=0.9,
                        weight_decay=0.0005)

criterion = nn.CrossEntropyLoss()

trainloader = torch.utils.data.DataLoader(yeast_dataset, batch_size=3,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(yeast_dataset, batch_size=3,
                                         shuffle=False, num_workers=0)

classes = ('background','cell')
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        ####inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
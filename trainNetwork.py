import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy


optimizer = optim.SGD(net.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=0.0005)

criterion = nn.CrossEntropyLoss()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('background','cell')
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
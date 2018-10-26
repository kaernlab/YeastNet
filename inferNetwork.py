import torch

import processImages as pi
from defineNetwork import Net
import validateNetwork as valNet
from processImages import YeastSegmentationDataset
from weightedLoss import WeightedCrossEntropyLoss

## Make Inference DataLoader
samplingList = list(range(pi.num_images()))
yeastDataset = YeastSegmentationDataset(samplingList, crop_size = 512)
inferenceLoader = torch.utils.data.DataLoader(yeastDataset, batch_size=1,
                                          shuffle=False, num_workers=0)

## Instantiate Net, Optimizer and Criterion
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = WeightedCrossEntropyLoss()

## Move Net to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

## Load Parameters and Optimizer
net.load_state_dict(torch.load("100epochmodel.pt"))
optimizer.load_state_dict(torch.load("model_opt.pt"))

## Inference 
loss = valNet.validate(net, device, inferenceLoader)
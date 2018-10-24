import torch

from defineNetwork import Net
import validateNetwork as valNet
from processImages import YeastSegmentationDataset

yeast_dataset = YeastSegmentationDataset(crop_size = 512)

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.load_state_dict(torch.load("100epochmodel.pt"))

loss = valNet.validate(net, device, yeast_dataset)
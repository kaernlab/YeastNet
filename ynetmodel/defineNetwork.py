import torch.nn as nn
from ynetmodel.netParts import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.group_size = 16
        self.input = inputConv(1,64, self.group_size)
        self.down1 = down(64,128, self.group_size)
        self.down2 = down(128,256, self.group_size)
        self.down3 = down(256,512, self.group_size)
        self.down4 = down(512,1024, self.group_size)
        self.up1 = middleUpConv(1024,512, self.group_size)
        self.up2 = up(1024,512, self.group_size)
        self.up3 = up(512,256, self.group_size)
        self.up4 = up(256,128, self.group_size)
        self.output = outputConv(128, 2, self.group_size)
 

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.output(x, x1)
        return x



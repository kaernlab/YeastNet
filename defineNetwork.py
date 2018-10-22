import torch.nn as nn
from netParts import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.input = inputConv(1,64)
        self.down1 = down(64,128)
        self.down2 = down(128,256)
        self.down3 = down(256,512)
        self.down4 = down(512,1024)
        self.up1 = middleUpConv(1024,512)
        self.up2 = up(1024,512)
        self.up3 = up(512,256)
        self.up4 = up(256,128)
        self.output = outputConv(128, 2)
 

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



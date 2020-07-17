import torch.nn as nn
import ynetmodel.netParts as newParts
import ynetmodel.netPartsOld as oldParts

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.group_size = 16
        self.input = newParts.inputConv(1,64, self.group_size)
        self.down1 = newParts.down(64,128, self.group_size)
        self.down2 = newParts.down(128,256, self.group_size)
        self.down3 = newParts.down(256,512, self.group_size)
        self.down4 = newParts.down(512,1024, self.group_size)
        self.up1 = newParts.middleUpConv(1024,512, self.group_size)
        self.up2 = newParts.up(1024,512, self.group_size)
        self.up3 = newParts.up(512,256, self.group_size)
        self.up4 = newParts.up(256,128, self.group_size)
        self.output = newParts.outputConv(128, 2, self.group_size)
 

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



class NetOld(nn.Module):
    def __init__(self):
        super(NetOld, self).__init__()

        self.input = oldParts.inputConv(1,64)
        self.down1 = oldParts.down(64,128)
        self.down2 = oldParts.down(128,256)
        self.down3 = oldParts.down(256,512)
        self.down4 = oldParts.down(512,1024)
        self.up1 = oldParts.middleUpConv(1024,512)
        self.up2 = oldParts.up(1024,512)
        self.up3 = oldParts.up(512,256)
        self.up4 = oldParts.up(256,128)
        self.output = oldParts.outputConv(128, 2)

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

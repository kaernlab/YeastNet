import torch
import torch.nn as nn
import torch.nn.functional as F


class down(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(input_depth,output_depth, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_depth,output_depth, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_depth,output_depth, 3, padding=1), #1024 -> 512
            nn.ReLU(inplace=True),
            nn.Conv2d(output_depth,output_depth, 3,padding=1), #512 -> 512
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(output_depth, output_depth // 2, 2, stride=2))

    def forward(self, x1, x2):
        x = torch.cat((x2, x1),1)
        x = self.conv(x)
        return x

class middleUpConv(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(middleUpConv, self).__init__()
        self.conv = nn.ConvTranspose2d(input_depth, output_depth, 2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x 


class inputConv(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(inputConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_depth,output_depth, 3, padding=1), #1->64
            nn.ReLU(inplace=True),
            nn.Conv2d(output_depth,output_depth, 3, padding=1), #64->64
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class outputConv(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(outputConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_depth, input_depth // 2, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(input_depth // 2,input_depth // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_depth // 2, output_depth, 1)) #64->2

    def forward(self, x1, x2):
        x = torch.cat((x2, x1),1)
        x = self.conv(x)
        return x

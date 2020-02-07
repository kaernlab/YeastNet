import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class down(nn.Module):
    def __init__(self, input_depth, output_depth, group_size = 16):
        super(down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2,2),
            #nn.Conv2d(input_depth,output_depth, 3, padding=1),
            Conv2d(input_depth,output_depth, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth),
            #nn.Conv2d(output_depth,output_depth, 3, padding=1),
            Conv2d(output_depth,output_depth, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, input_depth, output_depth, group_size = 16):
        super(up, self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(input_depth,output_depth, 3, padding=1),
            Conv2d(input_depth,output_depth, 3, padding=1), #1024 -> 512
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth),
            #nn.Conv2d(output_depth,output_depth, 3,padding=1), #512 -> 512
            Conv2d(output_depth,output_depth, 3,padding=1), 
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth),
            nn.ConvTranspose2d(output_depth, output_depth // 2, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth // 2)
        )

    def forward(self, x1, x2):
        x = torch.cat((x2, x1),1)
        x = self.conv(x)
        return x

class middleUpConv(nn.Module):
    def __init__(self, input_depth, output_depth, group_size = 16):
        super(middleUpConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(input_depth, output_depth, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth),
        )

    def forward(self, x):
        x = self.conv(x)
        return x 


class inputConv(nn.Module):
    def __init__(self, input_depth, output_depth, group_size = 16):
        super(inputConv, self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(input_depth,output_depth, 3, padding=1), #1->64
            Conv2d(input_depth,output_depth, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth),
            #nn.Conv2d(output_depth,output_depth, 3, padding=1), #64->64
            Conv2d(output_depth,output_depth, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, output_depth)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class outputConv(nn.Module):
    def __init__(self, input_depth, output_depth, group_size = 16):
        super(outputConv, self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(input_depth, input_depth // 2, 3, padding=1),
            Conv2d(input_depth, input_depth // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, input_depth // 2),
            #nn.Conv2d(input_depth // 2,input_depth // 2, 3, padding=1),
            Conv2d(input_depth // 2,input_depth // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(group_size, input_depth // 2),
            nn.Conv2d(input_depth // 2, output_depth, 1) #64->2
        )
    def forward(self, x1, x2):

        x = torch.cat((x2, x1),1)
        x = self.conv(x)
        return x

import numpy
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


#Utils functions
def load_image(timepoint):
    #Load all z stacks
    image = imageio.imread('Training Data/Images/z1_t_000_000_' + str(format(timepoint, '03d')) + '_BF.tif')
    image1 = imageio.imread('Training Data/Images/z2_t_000_000_' + str(format(timepoint, '03d')) + '_BF.tif')
    image2 = imageio.imread('Training Data/Images/z3_t_000_000_' + str(format(timepoint, '03d')) + '_BF.tif')
    #Rescale images to 0-1
    image = numpy.true_divide(image - image.min(), image.max() - image.min())
    image1 = numpy.true_divide(image1 - image1.min(), image1.max() - image1.min())
    image2 = numpy.true_divide(image2 - image2.min(), image2.max() - image2.min())
    #Stack the 3 zstacks into a 3 channels of an rgb image
    image3 = numpy.dstack((image,image1,image2))
    image3 = numpy.reshape(image3, (3,1040,1392))    
    x = image3[0:3, 264:-264, 440:-440]#.astype(double)
    return x
def show_image(image):
    #Display image
    plt.figure()
    plt.imshow(image)  
    plt.show()

def load_mask(timepoint):
    mask = sio.loadmat('Training Data/Masks/t_' + str(format(timepoint, '03d')) + '.mat')
    mask = (mask['LAB'] != 0)*1
    #x = mask[8:-8, 184:-184]#.astype(double)
    x = mask[264:-264, 440:-440]
    return x#.astype(double)
    
def num_images():
    return len(os.listdir('Training Data/Masks'))

def load_loss_map(timepoint):
    weightmap = numpy.load('Training Data/Loss Weight Maps/loss_weight_map' + str(timepoint) + '.npy')
    return weightmap[256:-256, 256:-256]


#Define Dataset Class
class YeastSegmentationDataset(Dataset):

    def __init__(self, transform=None):
        #self.segmentation_masks = [load_mask(i+1) for i in range(50)]
        #self.images = [load_image(i+1) for i in range(50)]
        self.transform = transform

    def __len__(self):
        length = num_images()
        return length

    def __getitem__(self,idx):
        idx+=1
        image = load_image(idx)
        mask = load_mask(idx)    
        #sample = {'image': image, 'mask': mask}
        
        weight_loss_matrix = load_loss_map(idx)
        sample = image, mask, weight_loss_matrix

        if self.transform:
            sample = self.transform(sample)

        return sample





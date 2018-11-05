import numpy
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pdb
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TV
import torchvision.transforms.functional as TF
#Utility functions


def numImages():
    return len(os.listdir('Training Data 1D/Images'))


#Define Dataset Class
class YeastSegmentationDataset(Dataset):

    def __init__(self, list_IDs, transform=None, crop_size = 512, random_rotate = False):
        self.ToTensor = TV.ToTensor()
        self.ToPILImage = TV.ToPILImage()
        self.crop_size = crop_size
        self.list_IDs = list_IDs
        self.all_data = list(range(numImages()))
        self.rotate = random_rotate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self,idx):
        data_id = self.all_data[self.list_IDs[idx]]
        mask = self.centerCrop(self.loadMask(data_id),self.crop_size)
        weight_loss_matrix = self.centerCrop(self.loadLossMap(data_id), self.crop_size)
        bw_image = self.loadImage(data_id)       
        bw_image = self.centerCrop(self.normalize(bw_image, bw_image.mean(), bw_image.std()), self.crop_size)

        if self.rotate:
            bw_image, mask, weight_loss_matrix = self.randomRotate(bw_image, mask, weight_loss_matrix)

        bw_image = self.ToTensor(bw_image)
        return bw_image, mask, weight_loss_matrix

    def centerCrop(self, image, new_size):
        if torch.is_tensor(image):
            c,h,w = image.shape
            image = image[:, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        else:
            h,w,c = image.shape
            image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2, :]
        return image

    def normalize(self, image, mean, std):
        image = (image - mean) / std
        image = (image - image.min()) / image.max()
        return image

    def loadImage(self, idx):
        image = imageio.imread('Training Data 1D/Images/im' + str(idx) + '.tif') 
        return image[:,:,numpy.newaxis].astype(numpy.double)

    def loadMask(self, idx):
        mask = sio.loadmat('Training Data 1D/Masks/mask' + str(idx) + '.mat')
        mask = (mask['LAB'] != 0)*1
        return mask[:,:,numpy.newaxis]
        
    def loadLossMap(self, idx):
        weightmap = numpy.load('Training Data 1D/Loss Weight Maps/lwm' + str(idx) + '.npy')
        return weightmap[:,:,numpy.newaxis].astype(numpy.float32)

    def showImage(self, image):
        #Display image
        plt.figure()
        plt.imshow(image)  
        plt.show()

    def randomRotate(self, bw_image, mask, loss_map):
        angle = random.choice([1, 2, 3, 4])
        for rotations in range(angle):
            bw_image = numpy.rot90(bw_image) - numpy.zeros_like(bw_image)
            mask = numpy.rot90(mask) - numpy.zeros_like(mask)
            loss_map = numpy.rot90(loss_map) - numpy.zeros_like(loss_map)
        
        return bw_image, mask, loss_map

    def random_sample(self, image, mask, weight_loss_matrix):
        x,y = int(numpy.random.randint((mask.shape[0]-161), size=1)), int(numpy.random.randint((mask.shape[0]-161), size=1))
        image = image[x:x+160,y:y+160]
        mask = mask[x:x+160,y:y+160]
        weight_loss_matrix = weight_loss_matrix[x:x+160,y:y+160]
        return image, mask, weight_loss_matrix
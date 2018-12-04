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

#Define Dataset Class
class YeastSegmentationDataset(Dataset):

    def __init__(self, list_IDs, transform=None, crop_size = 512, random_rotate = False, random_flip = False):
        self.ToTensor = TV.ToTensor()
        self.ToPILImage = TV.ToPILImage()
        self.crop_size = crop_size
        self.list_IDs = list_IDs
        self.all_data = list(range(len(os.listdir('Training Data 1D/Images'))))
        self.rotate = random_rotate
        self.flip = random_rotate

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
        if self.flip:
            bw_image, mask, weight_loss_matrix = self.randomFlip(bw_image, mask, weight_loss_matrix)

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
        image = imageio.imread('Training Data 1D/Images/im%03d.tif' % idx) 
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

    def random_sample(self, bw_image, mask, loss_map):
        x,y = int(numpy.random.randint((mask.shape[0]-161), size=1)), int(numpy.random.randint((mask.shape[0]-161), size=1))
        bw_image = bw_image[x:x+160,y:y+160]
        mask = mask[x:x+160,y:y+160]
        loss_map = loss_map[x:x+160,y:y+160]
        return bw_image, mask, loss_map

    def randomFlip(self, bw_image, mask, loss_map):
        to_flipud =  random.choice([0,1])
        to_fliplr =  random.choice([0,1])

        if to_fliplr:
            bw_image = numpy.fliplr(bw_image)
            mask = numpy.fliplr(mask)
            loss_map = numpy.fliplr(loss_map)

        if to_flipud:
            bw_image = numpy.flipud(bw_image)
            mask = numpy.flipud(mask)
            loss_map = numpy.flipud(loss_map)

        return bw_image, mask, loss_map
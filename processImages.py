import numpy
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pdb

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


#Utility functions
def load_image(idx):
    image = imageio.imread('Training Data 1D/Images/im' + str(idx) + '.tif') 
    return image[:,:,numpy.newaxis].astype(numpy.double)

def show_image(image):
    #Display image
    plt.figure()
    plt.imshow(image)  
    plt.show()

def load_mask(idx):
    mask = sio.loadmat('Training Data 1D/Masks/mask' + str(idx) + '.mat')
    mask = (mask['LAB'] != 0)*1
    return mask[:,:,numpy.newaxis]
    
def centre_crop(image, new_size):
    if torch.is_tensor(image):
        c,h,w = image.shape
        image = image[:, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
    else:
        h,w,c = image.shape
        image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2, :]
    return image

def num_images():
    return len(os.listdir('Training Data 1D/Images'))

def load_loss_map(idx):
    weightmap = numpy.load('Training Data 1D/Loss Weight Maps/lwm' + str(idx) + '.npy')
    return weightmap[:,:,numpy.newaxis].astype(numpy.float32)

def random_sample(image, mask, weight_loss_matrix):
    x,y = int(numpy.random.randint((mask.shape[0]-161), size=1)), int(numpy.random.randint((mask.shape[0]-161), size=1))
    image = image[x:x+160,y:y+160]
    mask = mask[x:x+160,y:y+160]
    weight_loss_matrix = weight_loss_matrix[x:x+160,y:y+160]
    return image, mask, weight_loss_matrix

def normalize_grayscale(image, mean, std):
    image = (image - mean) / std
    image = (image - image.min()) / image.max()
    return image

#Define Dataset Class
class YeastSegmentationDataset(Dataset):

    def __init__(self, list_IDs, transform=None, crop_size = 512):
        self.CenterCrop = centre_crop
        self.Normalize = normalize_grayscale
        self.ToTensor = torchvision.transforms.ToTensor()
        self.crop_size = crop_size
        self.list_IDs = list_IDs
        #self.num_images = num_images()
        self.all_data = list(range(num_images()))

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self,idx):
        data_id = self.all_data[self.list_IDs[idx]]
        mask = self.CenterCrop(load_mask(data_id),self.crop_size)
        weight_loss_matrix = self.CenterCrop(load_loss_map(data_id), self.crop_size)
        image = self.ToTensor(load_image(data_id))
        image = self.CenterCrop(self.Normalize(image, image.mean(), image.std()), self.crop_size)
        
        return image, mask, weight_loss_matrix





import numpy
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


#Utils functions
def load_image(timepoint, z_stack):
    title = 'Training Data/Images/z' + str(z_stack) + '_t_000_000_' + str(format(timepoint, '03d')) + '_BF.tif'
    image = imageio.imread(title) 
    return image[:,:,numpy.newaxis].astype(numpy.double)

def show_image(image):
    #Display image
    plt.figure()
    plt.imshow(image)  
    plt.show()

def load_mask(timepoint):
    mask = sio.loadmat('Training Data/Masks/t_' + str(format(timepoint, '03d')) + '.mat')
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
    return len(os.listdir('Training Data/Images'))

def load_loss_map(timepoint):
    weightmap = numpy.load('Training Data/Loss Weight Maps/loss_weight_map' + str(timepoint) + '.npy')
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

    def __init__(self, transform=None, crop_size = 512):
        self.CenterCrop = centre_crop
        self.Normalize = normalize_grayscale#torchvision.transforms.Normalize([0.5], [0.5])
        self.ToTensor = torchvision.transforms.ToTensor()
        self.crop_size = crop_size

    def __len__(self):
        return num_images()

    def __getitem__(self,idx):
        z_stack = idx // 51 + 1
        idx+=1
        if (idx % 51 == 0):
            timepoint = 51
        else:
            timepoint = idx % 51
        mask = self.CenterCrop(load_mask(timepoint),self.crop_size)
        weight_loss_matrix = self.CenterCrop(load_loss_map(timepoint), self.crop_size)
        image = self.ToTensor(load_image(timepoint, z_stack))
        image = self.CenterCrop(self.Normalize(image, image.mean(), image.std()), self.crop_size)
        
        return image, mask, weight_loss_matrix





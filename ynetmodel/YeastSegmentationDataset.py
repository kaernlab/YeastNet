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

    def __init__(self, list_IDs, transform=None, crop_size = 512, random_rotate = False,
                    random_flip = False, random_crop = False, no_og_data = False, normtype = 3, setMoments = False):
        self.ToTensor = TV.ToTensor()
        self.crop_size = crop_size
        self.list_IDs = list_IDs
        self.to_rotate = random_rotate
        self.to_flip = random_flip
        self.to_crop = random_crop
        self.no_og_data = no_og_data
        self.normtype = normtype

        if setMoments == False:
            self.setMoments = self.getDataSetMoments()
        else:
            self.setMoments = setMoments

    def __len__(self):
        return sum([len(x) for x in self.list_IDs.values()])

    def __getitem__(self,test_ID):

        dataset_names = list(self.list_IDs) # should be ['DSDataset],[YITDataset1],[YITDataset3]
        true_ID = test_ID
        true_dataset = dataset_names[0]

        for current_dataset, next_dataset in zip(dataset_names[:-1], dataset_names[1:]):
            dataset_size = len(self.list_IDs[current_dataset])
            if true_ID >= dataset_size:
                true_ID -= dataset_size
                true_dataset = next_dataset
            else:
                break
        true_ID = self.list_IDs[true_dataset][true_ID]
        data_ID = (true_ID, true_dataset)
        mask = self.loadMask(data_ID)
        weight_loss_matrix = self.loadLossMap(data_ID)
        bw_image = self.loadImage(data_ID)       
        bw_image = self.normalize(bw_image, true_dataset)

        if self.to_rotate:
            bw_image, mask, weight_loss_matrix = self.randomRotate(bw_image, mask, weight_loss_matrix, self.no_og_data)
        if self.to_flip:
            bw_image, mask, weight_loss_matrix = self.randomFlip(bw_image, mask, weight_loss_matrix)
        if self.to_crop:
            bw_image, mask, weight_loss_matrix = self.randomCrop(bw_image, mask, weight_loss_matrix, self.crop_size)

        #self.showImage(bw_image[:,:,0], weight_loss_matrix[:,:,0])

        bw_image = self.ToTensor(bw_image)
        return bw_image, mask, weight_loss_matrix

    def getDataSetMoments(self):
        setMoments = {key:{} for key in self.list_IDs}
        for dataset_name in list(self.list_IDs):
            mean = numpy.array([])
            std = numpy.array([])
            for imageID in self.list_IDs[dataset_name]:
                bw_image = imageio.imread('./Datasets/' + dataset_name + '/Images/im%03d.tif' % imageID)
                mean = numpy.append(mean, bw_image.mean())
                std = numpy.append(std, bw_image.std())

            setMoments[dataset_name]['mean'] = mean.mean()
            setMoments[dataset_name]['std'] = std.mean()

        return setMoments


    def centerCrop(self, image, new_size):
        if torch.is_tensor(image):
            _,h,w = image.shape
            image = image[:, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        else:
            h,w,_ = image.shape
            image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2, :]
        return image

    def normalize(self,image, dataset_name):
        #Per Database Norm 
        if self.normtype==2 or self.normtype==3:
            image = (image - self.setMoments[dataset_name]['mean']) / self.setMoments[dataset_name]['std']

        #Per Image Norm 
        if self.normtype==5:
            image = (image - image.mean())
            image = (image / image.std())

        #Rescaling with negative values
        if self.normtype==3:
            image = image + abs(image.min())
            image = image / image.max()

        if self.normtype==4:
            image = image - abs(image.min())
            image = image / image.max()

        #Old and incorrect method
        if self.normtype==1:
            image = (image - image.mean()) / image.std()
            image = (image - image.min()) / image.max()
        return image

    def loadImage(self, dataID):
        image = imageio.imread('./Datasets/' + dataID[1] + '/Images/im%03d.tif' % dataID[0]) 
        return image[:,:,numpy.newaxis].astype(numpy.double)

    def loadMask(self, dataID):
        mask = numpy.load('./Datasets/' + dataID[1] + '/Masks/mask%03d.npy' % dataID[0]) 
        mask = (mask != 0)*1
        return mask[:,:,numpy.newaxis]
        
    def loadLossMap(self, dataID):
        weightMap = numpy.load('./Datasets/' + dataID[1] + '/LossWeightMaps/lwm%03d.npy' % dataID[0]) 
        return weightMap[:,:,numpy.newaxis].astype(numpy.float32)

    def showImage(self, image, loss):
        #Display image
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(loss)
        plt.show()

    def randomRotate(self, bw_image, mask, loss_map, no_og_data):
        if no_og_data:
            angle = random.choice([2, 3, 4])
        else:
            angle = random.choice([1, 2, 3, 4])

        for rotations in range(angle):
            bw_image = numpy.rot90(bw_image) - numpy.zeros_like(bw_image)
            mask = numpy.rot90(mask) - numpy.zeros_like(mask)
            loss_map = numpy.rot90(loss_map) - numpy.zeros_like(loss_map)
        
        return bw_image, mask, loss_map

    def randomCrop(self, bw_image, mask, loss_map, crop_size):
        if mask.shape[0] > crop_size:
            x,y = int(numpy.random.randint((mask.shape[0]-(crop_size+1)), size=1)), int(numpy.random.randint((mask.shape[1]-(crop_size+1)), size=1))
            bw_image = bw_image[x:x+crop_size,y:y+crop_size]
            mask = mask[x:x+crop_size,y:y+crop_size]
            loss_map = loss_map[x:x+crop_size,y:y+crop_size]
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

        return numpy.ascontiguousarray(bw_image), numpy.ascontiguousarray(mask), numpy.ascontiguousarray(loss_map)
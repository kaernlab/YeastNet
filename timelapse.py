import imageio as imio
import numpy as np
import torch
import torchvision as tv
import os
import pdb
import scipy.spatial.distance as scipyD
import scipy.optimize as scipyO


class Timelapse():
    def __init__(self, device, image_dir = "inference"):
        self.device = device
        self.toTensor = tv.transforms.ToTensor()
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        self.num_images = len(self.image_filenames)
        self.total_cells = 0

        self.tensorsBW = [None] * self.num_images
        self.imagesBW = [None] * self.num_images#np.array([self.num_images])
        self.masks = [None] * self.num_images#np.array([self.num_images])
        self.labels = [None] * self.num_images#np.array([self.num_images])
        self.centroids = [None] * self.num_images#np.array([self.num_images])
        self.identity = [None] * self.num_images

    def loadImages(self, dimensions = 1024, normalize = False):
        for idx, image_name in enumerate(self.image_filenames):
            path = self.image_dir + '/' + image_name
            imageBW = imio.imread(path) 

            if normalize:
                imageBW = self.normalizeGrayscale(imageBW, imageBW.mean(), imageBW.std())

            tensorBW = imageBW[:,:,np.newaxis].astype(np.double)
            tensorBW = self.toTensor(tensorBW).unsqueeze_(0).float()
            tensorBW = self.centreCrop(tensorBW.to(self.device), dimensions)

            self.tensorsBW[idx] = tensorBW
            self.imagesBW[idx] = self.centreCrop(imageBW, dimensions)

    def makeMasks(self, masks):
        for idx, mask in enumerate(masks):
            bgMask = mask.cpu().detach().numpy()[0,0,:,:]
            cellMask = mask.cpu().detach().numpy()[0,1,:,:]
            mask = np.zeros(bgMask.shape)
            mask = (cellMask > bgMask) * 1
            self.masks[idx] = mask

    def centreCrop(self, image, new_size):
        h,w = image.shape[-2:]
        if len(image.shape) > 2:
            image = image[:, :, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        else:
            image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        return image

    def normalizeGrayscale(self, image, mean, std):
        image = (image - mean) / std
        image = (image - image.min()) / image.max()
        return image

    def cellTrack(self):
        self.identity[0] = np.unique(self.labels[0])[:]
        self.total_cells = len(self.identity[0])

        for idx, (first, second) in enumerate(zip(self.centroids[:-1], self.centroids[1:])):
            timepoint = idx+1
            Y = scipyD.cdist(first, second, 'euclidean')
            firstLabels, secondLabels = scipyO.linear_sum_assignment(Y)
            firstLabels = self.identity[timepoint-1]
            self.identity[timepoint] = np.full(len(second), -1)
            for idx, label in enumerate(secondLabels):
                self.identity[timepoint][label] = firstLabels[idx]
            #pdb.set_trace()
            for idx, label in enumerate(self.identity[timepoint]):
                if label == -1:
                    self.identity[timepoint][idx] = self.total_cells
                    self.total_cells += 1
            

            

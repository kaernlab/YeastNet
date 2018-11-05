import imageio as imio
import numpy as np
import torch
import torchvision as tv


class timelapse():
    def __init__(self, device):
        self.device = device
        self.toTensor = tv.transforms.ToTensor()
        self.labels = 0

    def load_image(self, path, dimension = 1024, normalize = False):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        imageBW = imio.imread(path) 
        if normalize:
            imageBW = self.normalize_grayscale(imageBW, imageBW.mean(), imageBW.std())

        tensorBW = imageBW[:,:,np.newaxis].astype(np.double)
        tensorBW = self.toTensor(tensorBW).unsqueeze_(0).float()
        tensorBW = self.centre_crop(tensorBW.to(self.device), dimension)

        self.tensorBW = tensorBW
        self.imageBW = self.centre_crop(imageBW, dimension)

    def make_mask(self, masks):
        bgMask = masks.cpu().detach().numpy()[0,0,:,:]
        cellMask = masks.cpu().detach().numpy()[0,1,:,:]
        mask = np.zeros(bgMask.shape)
        mask = (cellMask > bgMask) * 1
        self.mask = mask

    def centre_crop(self, image, new_size):
        h,w = image.shape[-2:]
        if len(image.shape) > 2:
            image = image[:, :, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        else:
            image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        return image

    def normalize_grayscale(self, image, mean, std):
        image = (image - mean) / std
        image = (image - image.min()) / image.max()
        return image
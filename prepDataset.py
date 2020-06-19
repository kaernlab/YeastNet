import numpy as np
import imageio
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pdb
import random
import cv2


def crop(image):

    return image[7:1031, 355:1379]

x = [4,5,6,7,8,9,10,11,12,13,14,15,25,26,27,46,47,48,49,50]


for idx, imageID in enumerate(x):
    mask_path = 'C:/Users/Salemd/Desktop/xy3masks/' + 'Mask{:03d}.tif'.format(imageID)
    image1_path = 'C:/Users/Salemd/Desktop/New folder/' + 'z1_t_000_000_{:03d}_BF.tif'.format(imageID)
    image2_path = 'C:/Users/Salemd/Desktop/New folder/' + 'z2_t_000_000_{:03d}_BF.tif'.format(imageID)
    image3_path = 'C:/Users/Salemd/Desktop/New folder/' + 'z3_t_000_000_{:03d}_BF.tif'.format(imageID)
    mask = imageio.imread(mask_path)
    image1 = imageio.imread(image1_path)
    image2 = imageio.imread(image2_path)
    image3 = imageio.imread(image3_path)
    mask = crop(mask)
    image1 = crop(image1)
    image2 = crop(image2)
    image3 = crop(image3)

    mask = cv2.connectedComponents(mask, 4, cv2.CV_32S)

    np.save('./Datasets/DSDataset2/Masks' + '/mask{:03d}'.format(idx), mask[1])
    np.save('./Datasets/DSDataset2/Masks' + '/mask{:03d}'.format(idx+20), mask[1])
    np.save('./Datasets/DSDataset2/Masks' + '/mask{:03d}'.format(idx+40), mask[1])

    imageio.imwrite('./Datasets/DSDataset2/Images' + '/im{:03d}.tif'.format(idx), image1)
    imageio.imwrite('./Datasets/DSDataset2/Images' + '/im{:03d}.tif'.format(idx+20), image2)
    imageio.imwrite('./Datasets/DSDataset2/Images' + '/im{:03d}.tif'.format(idx+40), image3)
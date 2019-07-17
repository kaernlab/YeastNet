import numpy as np
import scipy.io as sio
from scipy import ndimage
import imageio
import pdb
import matplotlib.pyplot as plt


def load_mask1(timepoint):
    mask = sio.loadmat('./Training Data/Masks/mask%01d.mat' % timepoint)
    cropped_mask = mask['LAB'][8:-8, 184:-184]
    return cropped_mask

def load_mask2(timepoint, dataset):
    image = imageio.imread('./Datasets/' + dataset + '/Loss Weight Maps/mask%03d.tif' % timepoint)
    #image = image[8:-8, 184:-184]
    return image

def getLossMatrix(imageID, dataset):

    #gt = load_mask2(imageID, dataset)
    gt = np.load('./Datasets/' + dataset + '/Masks/mask%03d.npy' % imageID)
    gt2 = ~(gt==0)
    uvals=np.unique(gt2)
    wmp=np.zeros(uvals.shape)

    wmp = [1/np.sum(gt2==uvals[uv]) for uv in range(uvals.shape[0])]

    wmp=wmp / np.max(wmp)

    wc=np.zeros(gt.shape)
    for uv in range(uvals.shape[0]):
        wc[gt2==uvals[uv]]=wmp[uv]
    w0 = 10 
    sigma = 10

    num_cells = np.max(gt)

    bwgt=np.zeros(gt.shape)
    maps=np.zeros((gt.shape[0],gt.shape[1],num_cells))
    
    if num_cells>=2:
        for cellID in range(num_cells):
            maps[:,:,cellID]=ndimage.distance_transform_edt(~(gt == cellID+1))

        maps.sort(axis=2)
        d1=maps[:,:,0]
        d2=maps[:,:,1]

        bwgt=w0 * np.exp((-(np.power((d1+d2),2))) / (2*sigma) ) * (gt==0)
        weight = wc + bwgt
    #plt.imshow(bwgt)
    #plt.show()
    #plt.imshow(weight)
    #plt.show()
    #plt.imshow(wc)
    #plt.show()

    return weight

dataset = 'YITDataset3'
timepoints = 20
for idx in range(timepoints):
    weight = getLossMatrix(idx, dataset)
    np.save(('./Datasets/' + dataset + '/LossWeightMaps/lwm%03d' % idx), weight)
    #image = np.load('./Datasets/' + dataset + '/Loss Weight Maps/lwm%01d.npy' % idx)
    imageio.imwrite(('./Datasets/' + dataset + '/New/lwm%03d.png' % idx), weight)

#for idx in list(range(153)):
    #imageio.imwrite(('./Datasets/YITDataset3/New/mask%03d.tif' % idx), load_image(idx))
#    np.save(('./Datasets/DSDataset/New/mask%03d' % idx), load_image(idx))
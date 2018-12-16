import numpy as np
import scipy.io as sio
from scipy import ndimage
import matplotlib.pyplot as plt


def load_mask(timepoint):
    mask = sio.loadmat('Training Data/Masks/t_' + str(format(timepoint, '03d')) + '.mat')
    x = mask['LAB'][8:-8, 184:-184]
    return x

def getLossMatrix(imageID):

    gt = load_mask(imageID)
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
    #show_image(bwgt)
    #show_image(weight)
    #show_image(wc)

    return weight

#for id in range(1,51):
weight = getLossMatrix(51)
print(weight)
np.save('loss_weight_map' + str(51), weight)
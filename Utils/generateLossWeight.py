import numpy as np
import scipy.io as sio
from scipy import ndimage
import imageio
import pdb
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Produce Loss Matrix
def getLossMatrix(imageID, dataset, w0 = 10, sigma = 10):

    # Load Ground Truth Mask
    gt = np.load('../Datasets/' + dataset + '/Masks/mask%03d.npy' % imageID)
    gt2 = ~(gt==0)
    uvals=np.unique(gt2)
    wmp=np.zeros(uvals.shape)

    wmp = [1/np.sum(gt2==uvals[uv]) for uv in range(uvals.shape[0])]
    
    wmp=wmp / np.max(wmp)

    wc=np.zeros(gt.shape)
    for uv in range(uvals.shape[0]):
        wc[gt2==uvals[uv]]=wmp[uv]

    num_cells = np.max(gt)

    bwgt=np.zeros(gt.shape)
    maps=np.zeros((gt.shape[0],gt.shape[1],num_cells))
    if num_cells>=2:
        for cellID in range(num_cells):
            maps[:,:,cellID]=ndimage.distance_transform_edt(~(gt == cellID+1))

        maps.sort(axis=2)
        d1=maps[:,:,0]
        d2=maps[:,:,1]
        bwgt = w0 * np.exp((-(np.power((d1+d2),2))) / np.power((2*sigma), 2)) * (gt==0)
        weight = wc + bwgt

    return weight


if __name__ == "__main__":
    ## Parameters
    datasets = ['DSDataset','YITDataset1','YITDataset3']
    dataset = datasets[0]
    w0 = 20 # Absolute scaling
    sigma = 10 # Higher means further pixels have greater weight
    
    ## Set Number of Timepoints in Dataset
    if dataset == 'YITDataset3':
        timepoints = 20
    elif dataset == 'YITDataset1':
        timepoints = 60
    else:
        timepoints = 51

    folder_name1 = '../Datasets/' + dataset + '/LossWeightMaps/{}.{}'.format(w0,sigma)
    if not os.path.exists(folder_name1):
        os.mkdir(folder_name1)
    folder_name2 = '../Datasets/' + dataset + '/LossWeightImages/{}.{}'.format(w0,sigma)
    if not os.path.exists(folder_name2):
        os.mkdir(folder_name2)

    for idx in tqdm(range(timepoints)):
        weight = getLossMatrix(idx, dataset, w0 = w0, sigma = sigma)
        np.save(folder_name1 + '/lwm{:03d}'.format(idx), weight)
        if dataset == 'DSDataset':
            np.save(folder_name1 + '/lwm{:03d}'.format(idx+51), weight)
            np.save(folder_name1 + '/lwm{:03d}'.format(idx+102), weight)

        imageio.imwrite(folder_name2 + '/lwm{:03d}.png'.format(idx), np.uint8(weight))
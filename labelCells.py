import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pdb
import cv2
from skimage.morphology import watershed

import processImages as pi


def load_image(path):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    image = imageio.imread(path) 
    return image

def label_cells(mask, bw_image):

    ## Loads a predicted mask and detects individual cells in the image.
    #image = load_image(image_path + 'Pred.png')
    #bw_image = load_image(image_path + 'IMG.png')
    bw_image = np.dstack((bw_image,bw_image,bw_image))
    #bw_image = np.reshape(bw_image, (512,512,3)) 

    kernel3 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((2,2),np.uint8)
    labels = np.array(mask.size)
    image2 = cv2.erode(mask,kernel3,iterations = 3)
    image3 = cv2.dilate(image2,kernel3,iterations = 3)

    dist_transform = cv2.distanceTransform(image3,cv2.DIST_L2,5)
    sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
    sure_fg = cv2.dilate(sure_fg[1].astype(np.uint8), kernel3 ,iterations = 3)
    #sure_fg = cv2.dilate(sure_fg,kernel3,iterations = 3)
    #pi.show_image(sure_fg[1])

    output = cv2.connectedComponentsWithStats(sure_fg, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
    #output = cv2.connectedComponentsWithStats(image3, 4, cv2.CV_32S) 
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    markers = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
 
    
    labels = watershed(-dist_transform, markers, mask=image3)
    labels = np.array(labels)
    ## Get label Areas


    contours = []
    areas = []
    for idx in range(1,num_labels):
        single_label_mask = (labels==idx).astype(np.uint8)
        im2, contour, hierarchy = cv2.findContours(single_label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(contour)
        contouredImage = cv2.drawContours(bw_image, contour, -1, (0,0,255), 1)
        areas.append(single_label_mask.sum())
    
    return centroids[1:], contouredImage, labels, areas
    #fig, axs = plt.subplots(1, 2)
    #axs[0].imshow(labels) 
    #axs[1].imshow(x)
    #for idx, cnt in enumerate(centroids[1:]):
    #    axs[1].text(cnt[0]-2, cnt[1]+2, str(idx+1), fontsize=8, color='r')
    #plt.show()

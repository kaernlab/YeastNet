import imageio
import numpy as np
import pdb
import cv2
from skimage.morphology import watershed
from Utils.helpers import load_image

def label_cells(mask, bw_image):

    ## Loads a predicted mask and detects individual cells in the image.
    bw_image = np.dstack((bw_image,bw_image,bw_image)) 

    kernel3 = np.ones((3,3), np.uint8)
    labels = np.array(mask.size)
    mask = cv2.erode(mask, kernel3, iterations = 3)
    clean_mask = cv2.dilate(mask, kernel3, iterations = 3)

    fg_threshold = 0.455
    dist_transform = cv2.distanceTransform(clean_mask,cv2.DIST_L2,5)
    sure_fg = cv2.threshold(dist_transform,fg_threshold*dist_transform.max(),255,0)
    sure_fg = cv2.dilate(sure_fg[1].astype(np.uint8), kernel3 ,iterations = 3)

    output = cv2.connectedComponentsWithStats(sure_fg, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT) 
    num_labels = output[0]
    markers = output[1]
    #stats = output[2]
    centroids = output[3]

    labels = watershed(-dist_transform, markers, mask=clean_mask)
    labels = np.array(labels)

    ## Get label Areas
    contours = []
    areas = []
    for idx in range(1,num_labels):
        single_label_mask = (labels==idx).astype(np.uint8)
        _, contour, _ = cv2.findContours(single_label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours.append(contour)
        contouredImage = cv2.drawContours(bw_image, contour, -1, (0,0,255), 1)
        areas.append(single_label_mask.sum())
    
    return centroids[1:], contouredImage, labels, areas

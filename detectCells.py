import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import pdb
import cv2

import processImages as pi


def load_image(path):
    image = imageio.imread(path) 
    return image

## Loads a predicted mask and detects individual cells in the image.
image_path = './Validation/6Pred.png'
image = load_image(image_path)
kernel = np.ones((3,3),np.uint8)
labels = np.array(image.size)

image2 = cv2.erode(image,kernel,iterations = 3)
image3 = cv2.dilate(image2,kernel,iterations = 3)

#pi.show_image(image)
#pi.show_image(image2)
#pi.show_image(image3)

dist_transform = cv2.distanceTransform(image,cv2.DIST_L2,5)

sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)

pi.show_image(sure_fg[1])

markers = cv2.connectedComponents(sure_fg[1].astype(np.uint8), 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)

pi.show_image(markers[1])


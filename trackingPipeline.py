##
#import matplotlib.pyplot as plt

import torchvision as tv
import numpy as np
import torch
import pdb
import imageio
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2
import scipy.spatial.distance as scipyD
import scipy.optimize as scipyO
## 
import inferNetwork
import labelCells
from timelapse import Timelapse

import numpy

def smooth(x,window_len=5,window='flat'):
    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')
    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y




##
def makeTL():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tl = Timelapse(device = device, image_dir = 'Test')

    # Load image for inference 
    tl.loadImages(normalize = True, dimensions = 1024)
    # Pass Image to Inference script, return predicted Mask
    predictions = inferNetwork.inferNetworkBatch(images = tl.tensorsBW, num_images = tl.num_images, device = device)
    tl.makeMasks(predictions)

    for idx, mask in enumerate(tl.masks):
        imageio.imwrite(tl.image_dir + '/Results/' + str(idx) + 'Pred.png', mask)

    # Pass Mask into cell labeling script, return labelled cells 
    for idx, (imageBW, mask) in enumerate(zip(tl.imagesBW, tl.masks)):
        tl.centroids[idx], tl.contouredImages[idx], tl.labels[idx], tl.areas[idx] = labelCells.label_cells(np.array(mask), np.array(imageBW))
        imageio.imwrite(tl.image_dir + '/Results/' + str(idx) + 'Labels.png', tl.labels[idx])
        imageio.imwrite(tl.image_dir + '/Results/' + str(idx) + 'Overlay.png', tl.contouredImages[idx])

    tl.cellTrack()

    tl.DrawTrackedCells()

    with open('./inference/timelapse.pkl', 'wb') as f:
        pickle.dump(tl, f, pickle.HIGHEST_PROTOCOL)

makeTL()


with open('./inference/timelapse.pkl', 'rb') as f:
    tl = pickle.load(f)

for i in range(tl.num_images):
    mask = sio.loadmat(tl.image_dir + '/Masks/mask' + str(i))
    mask = mask['LAB']
    mask = tl.centreCrop(mask, 1024)
    output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
    num_labels = output[0]
    markers = output[1]
    stats = output[2]
    centroids = output[3]

    #print(tl.centroids[1])
    #print(centroids[1:])
    predCent = tl.centroids[i]
    trueCent = centroids[1:]
    centroidDiff = scipyD.cdist(predCent, trueCent, 'euclidean')
    firstLabels, secondLabels = scipyO.linear_sum_assignment(centroidDiff)

    accurateSeg = 0
    trueSeg = len(trueCent)
    for pred, true  in zip(firstLabels,secondLabels):

        print(centroidDiff[pred, true])
        if centroidDiff[pred, true] < 5 :
            accurateSeg += 1
        
    print(accurateSeg, trueSeg)
pdb.set_trace()


plt.figure()
for i in range(tl.total_cells):
    x, gfpfl, rfpfl = tl[i]
    if len(x)>9:
        gfpfl = smooth(gfpfl)
        rfpfl = smooth(rfpfl)
        diff = int(abs(len(x) - len(gfpfl)) / 2)
        #print(diff)
        gfpfl = gfpfl[diff:-diff]
        rfpfl = rfpfl[diff:-diff]
        ratio = rfpfl / gfpfl 
        #ratio = ratio / ratio.max()
        plt.plot(x,ratio)

plt.show()
##
#import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import imageio
import pickle
import matplotlib.pyplot as plt
import cv2
import argparse
import scipy.io as sio
import os
## 
import inferNetwork
import labelCells
from Timelapse import Timelapse
from Utils.helpers import smooth
from Utils.helpers import accuracy
from Utils.helpers import centreCrop
from TestPerformance import testMeasureF

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", type=str, help="input string of image directory", default="Test")
args = parser.parse_args()

imagedir = args.imagedir

##
def makeTL(imagedir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tl = Timelapse(device = device, image_dir = imagedir)

    # Load image for inference 
    tl.loadImages(normalize = True, dimensions = 1024, toCrop = True)
    # Pass Image to Inference script, return predicted Mask
    predictions = inferNetwork.inferNetworkBatch(images = tl.tensorsBW, num_images = tl.num_images, device = device)
    tl.makeMasks(predictions)

    # Make folder if doesnt exist
    if not os.path.isdir(tl.image_dir + 'Results'):
        os.mkdir(tl.image_dir + 'Results')
        os.mkdir(tl.image_dir + 'Results/YeastNet')
        os.mkdir(tl.image_dir + 'Results/Tracking')

    for idx, mask in enumerate(tl.masks):
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'Pred.png', mask)

    # Pass Mask into cell labeling script, return labelled cells 
    for idx, (imageBW, mask) in enumerate(zip(tl.imagesBW, tl.masks)):
        tl.centroids[idx], tl.contouredImages[idx], tl.labels[idx], tl.areas[idx] = labelCells.label_cells(np.array(mask), np.array(imageBW))
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'Labels.png', tl.labels[idx])
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'Overlay.png', tl.contouredImages[idx])
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'BWimage.png', imageBW)

    tl.cellTrack()

    tl.DrawTrackedCells()

    with open(tl.image_dir + 'Results/timelapse.pkl', 'wb') as f:
        pickle.dump(tl, f, pickle.HIGHEST_PROTOCOL)

    return tl

def showTraces(tl):

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


def getAccuracy(tl):

    for idx, pred_mask in enumerate(tl.masks):

        true_mask = sio.loadmat('Training Data 1D/Masks/mask' + str(idx) + '.mat')
        true_mask = (true_mask['LAB_orig'] != 0)*1
        true_mask = centreCrop(true_mask, 1024)

        PixAccuracy, IntOfUnion = accuracy(true_mask, pred_mask)


        return IntOfUnion[1]

tl = makeTL(imagedir)
#with open(imagedir + 'Results/timelapse.pkl', 'rb') as f:
#    tl = pickle.load(f)
#testMeasureF(tl, makeTG=False)
print(getAccuracy(tl))

#showTraces(tl)
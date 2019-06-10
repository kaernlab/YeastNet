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
import shutil
## 
import inferNetwork
import labelCells
from Timelapse import Timelapse
from Utils.compareMethods import compareOld2
from Utils.compareMethods import makeAccCSV
from Utils.helpers import smooth
from Utils.helpers import accuracy
from Utils.helpers import centreCrop
from Utils.TestPerformance import makeResultsCSV

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", type=str, help="Input string of image directory", default="")
parser.add_argument("--model_num", type=str, help="Cross Validation model number to use", default="0")
parser.add_argument("--make_plot", type=str, help="Boolean variable to choose whether fluorescence plots are desired", default=False)
args = parser.parse_args()

imagedir = args.imagedir
model_num = args.model_num
makePlots = args.make_plot

imagedir = '../tracking-analysis-py/Data/stable_60/FOV4/BF/'
model_path = './TrackingTest/trackingtest.pt'
##
def makeTL(imagedir, crossval):
    #model_path = './CrossValidation/Finetuned Models/model_cp' + str(crossval) + '.pt'
    #model_path = './TrackingTest/trackingtest.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tl = Timelapse(device = device, image_dir = imagedir)

    # Load image for inference 
    tl.loadImages(normalize = True, dimensions = 1024, toCrop = True)
    # Pass Image to Inference script, return predicted Mask
    predictions = inferNetwork.inferNetwork(images = tl.tensorsBW, num_images = tl.num_images, device = device, model_path = model_path)
    tl.makeMasks(predictions)

    # Make folder if doesnt exist
    if not os.path.isdir(tl.image_dir + 'Results'):
        os.mkdir(tl.image_dir + 'Results')
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
    total = 0
    plt.figure()
    for i in range(20):#range(tl.total_cells):
        x, gfpfl, rfpfl = tl[i]
        #pdb.set_trace()
        if len(x)>30 and total<15 and i != 5:
            total+=1
            #gfpfl = smooth(gfpfl, window_len=5)
            #rfpfl = smooth(rfpfl, window_len=7)
            diff = int(abs(len(x) - len(gfpfl)) / 2)
            #print(diff)
            #gfpfl = gfpfl[diff:-diff]
            #rfpfl = rfpfl[diff:-diff]
            ratio = np.array(rfpfl) / np.array(gfpfl) 
            ratio = smooth(ratio, window_len=5)
            diff = int(abs(len(x) - len(ratio)) / 2)
            ratio = ratio[diff:-diff]
            #ratio = ratio / ratio.max()
            plt.plot(np.array(x) * 10,ratio, 'k')

    #pdb.set_trace()
    plt.ylabel('mCherry/sfGFP Fluorescence Ratio', fontsize=15)
    plt.xlabel('Time (minutes)', fontsize=15)
    plt.show()


def getMeanAccuracy(tl, model_num = 0):
    runningIOU = 0
    checkpoint = torch.load(model_path)
    testIDs = list(range(50))#checkpoint['testID']
    testIDs.sort()


    for testID, pred_mask in zip(testIDs, tl.masks):
        true_mask = sio.loadmat('Training Data 1D/Masks/mask' + str(testID) + '.mat')
        true_mask = (true_mask['LAB_orig'] != 0)*1
        true_mask = centreCrop(true_mask, 1024)
        #pdb.set_trace()
        pred_mask = (pred_mask != 0)*1
        PixAccuracy, IntOfUnion = accuracy(true_mask, pred_mask)

        runningIOU += IntOfUnion[1]

    
    return runningIOU / 50


tl = makeTL(imagedir, model_num)
with open(imagedir + 'Results/timelapse.pkl', 'rb') as f:
    tl = pickle.load(f)

if makePlots:
    showTraces(tl)

#makeResultsCSV(tl, makeTG=True)
#print(compareOld2(model_path))
#print(getMeanAccuracy(tl, model_num))
#makeAccCSV(model_path, imagedir)

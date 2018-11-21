##
#import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
import imageio
import pickle
import matplotlib.pyplot as plt
import cv2

## 
import inferNetwork
import labelCells
from timelapse import Timelapse
from Utils.helpers import smooth
from TestPerformance import testMeasureF

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

with open('./inference/timelapse.pkl', 'rb') as f:
    tl = pickle.load(f)

makeTL()
testMeasureF(tl)
#showTraces(tl)
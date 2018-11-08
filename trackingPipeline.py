##
#import matplotlib.pyplot as plt

import torchvision as tv
import numpy as np
import torch
import pdb
import imageio
import pickle

## 
import inferNetwork
import labelCells
from timelapse import Timelapse


##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tl = Timelapse(device = device, image_dir = 'inference')

# Load image for inference 
tl.loadImages(normalize = True, dimensions = 1024)
# Pass Image to Inference script, return predicted Mask
predictions = inferNetwork.inferNetworkBatch(images = tl.tensorsBW, num_images = tl.num_images, device = device)
tl.makeMasks(predictions)

for idx, mask in enumerate(tl.masks):
    imageio.imwrite('inference/Results/' + str(idx) + 'Pred.png', mask)

# Pass Mask into cell labeling script, return labelled cells 
for idx, (imageBW, mask) in enumerate(zip(tl.imagesBW, tl.masks)):
    tl.centroids[idx], tl.contouredImages[idx], tl.labels[idx], tl.areas[idx] = labelCells.label_cells(np.array(mask), np.array(imageBW))
    imageio.imwrite('inference/Results/' + str(idx) + 'Labels.png', tl.labels[idx])
    imageio.imwrite('inference/Results/' + str(idx) + 'Overlay.png', tl.contouredImages[idx])


tl.cellTrack()

tl.DrawTrackedCells()

with open('./inference/timelapse.pkl', 'wb') as f:
    pickle.dump(tl, f, pickle.HIGHEST_PROTOCOL)

##
import matplotlib.pyplot as plt
import torchvision as tv
import numpy as np
import torch
import pdb
import imageio

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
masks = inferNetwork.inferNetworkBatch(images = tl.tensorsBW, num_images = tl.num_images, device = device)
#pdb.set_trace()
tl.makeMasks(masks)
#imageio.imwrite('inference/Pred.png', tl.mask)

# Pass Mask into cell labeling script, return labelled cells 
for idx, (imageBW, mask) in enumerate(zip(tl.imagesBW, tl.masks)):
    tl.centroids[idx], x, tl.labels[idx] = labelCells.label_cells(np.array(mask).astype(np.uint8), np.array(imageBW))
    #imageio.imwrite('inference/label' + str(idx) + '.png', np.array(tl.labels[idx]))
    #imageio.imwrite('inference/overlay' + str(idx) + '.png', x)


tl.cellTrack()

for imageID in range(tl.num_images-1):
    plt.figure()
    for idx, (label, cnt) in enumerate(zip(tl.identity[imageID], tl.centroids[imageID])):
        plt.imshow(tl.imagesBW[imageID])
        plt.text(cnt[0]-2, cnt[1]+2, label, fontsize=8, color='r')
    plt.show()

# Test Image
#plt.figure()
#plt.imshow(x) 
#plt.show()


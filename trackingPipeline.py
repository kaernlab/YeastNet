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
predictions = inferNetwork.inferNetworkBatch(images = tl.tensorsBW, num_images = tl.num_images, device = device)
tl.makeMasks(predictions)

for idx, mask in enumerate(tl.masks):
    imageio.imwrite('inference/Results/' + str(idx) + 'Pred.png', mask)

# Pass Mask into cell labeling script, return labelled cells 
for idx, (imageBW, mask) in enumerate(zip(tl.imagesBW, tl.masks)):
    tl.centroids[idx], x, tl.labels[idx] = labelCells.label_cells(np.array(mask).astype(np.uint8), np.array(imageBW))
    imageio.imwrite('inference/Results/' + str(idx) + 'Labels.png', np.array(tl.labels[idx]))
    imageio.imwrite('inference/Results/' + str(idx) + 'Overlay.png', x)


tl.cellTrack()

for imageID in range(tl.num_images-1):
    fig = plt.figure()
    bw_image = np.dstack((tl.imagesBW[imageID],tl.imagesBW[imageID],tl.imagesBW[imageID]))
    for idx, (label, cnt) in enumerate(zip(tl.identity[imageID], tl.centroids[imageID])):
        plt.imshow(bw_image)
        plt.text(cnt[0]-2, cnt[1]+2, label, fontsize=6, color='r')
        
    fig.savefig('inference/Results/' + str(idx) + 'Tracked.png', bbox_inches='tight')
    plt.close(fig)
    ##plt.show()


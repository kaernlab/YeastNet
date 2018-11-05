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
from timelapse import timelapse


##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tl = timelapse(device = device)

# Load image for inference 
tl.load_image('./inference/test.tif', normalize = True, dimension = 1024)

# Pass Image to Inference script, return predicted Mask
masks = inferNetwork.inferNetworkSingle(tl.tensorBW)

tl.make_mask(masks)
#imageio.imwrite('inference/Pred.png', tl.mask)

# Pass Mask into cell labeling script, return labelled cells 
#pdb.set_trace()
x, tl.labels = labelCells.label_cells(tl.mask.astype(np.uint8), tl.imageBW)


# Test Image
plt.figure()
plt.imshow(tl.labels) 
plt.show()



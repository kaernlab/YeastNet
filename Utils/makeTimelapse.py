import numpy as np
import torch
import pdb
import imageio
import pickle
import argparse
import os

from ynetmodel.detect import infer
from Utils.labelCells import labelCells
from Utils.Timelapse import Timelapse

def makeTimelapse(imagedir, model_path, saveExp):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tl = Timelapse(device = device, image_dir = imagedir)

    # Load image for inference 
    tl.loadImages(normalize = True)
    
    # Pass Image to Inference script, return predicted Mask
    predictions = infer(images = tl.tensorsBW, num_images = tl.num_images, device = device, model_path = model_path)
    tl.makeMasks(predictions)

    # Make folder if doesnt exist
    if not os.path.isdir(tl.image_dir + 'Results'):
        os.mkdir(tl.image_dir + 'Results')
        os.mkdir(tl.image_dir + 'Results/Tracking')

    #Save images of predicted masks
    for idx, mask in enumerate(tl.masks):
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'Pred.png', mask)

    # Pass Mask into cell labeling script, return labelled cells. Save Images
    for idx, (imageBW, mask) in enumerate(zip(tl.imagesBW, tl.masks)): 
        tl.centroids[idx], tl.contouredImages[idx], tl.labels[idx], tl.areas[idx] = labelCells(np.array(mask), np.array(imageBW))
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'Labels.png', tl.labels[idx])
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'Overlay.png', (tl.contouredImages[idx] * 255).astype('uint8'))
        imageio.imwrite(tl.image_dir + 'Results/' + str(idx) + 'BWimage.png', (imageBW * 255).astype('uint8'))

    # Only conduct tracking if there is more than one image
    if tl.num_images > 1:
        tl.cellTrack()
        tl.DrawTrackedCells()

    if saveExp:
        # Save timelapse experiment in the Results folder
        with open(tl.image_dir + 'Results/timelapse.pkl', 'wb') as f:
            pickle.dump(tl, f, pickle.HIGHEST_PROTOCOL)

    return tl


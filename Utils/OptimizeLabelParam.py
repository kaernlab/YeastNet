import numpy as np
import scipy.optimize as optimize
import cv2
from skimage.morphology import watershed
from timelapse import Timelapse
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
from scipy import io
from defineNetwork import Net
import pdb
import imageio



def label_cells(predMask, x0):
    ## Loads a predicted mask and detects individual cells in the image.
    # Make Kernel and assign names to parameters
    kernel3 = np.ones((3,3),np.uint8)
    foreground_threshold = x0

    # Remove objects smaller than 3x3 pixels
    predMask = cv2.erode(predMask,kernel3,iterations = 3)
    predMask = cv2.dilate(predMask,kernel3,iterations = 3)

    # Find Watershed Markers for Cells 
    dist_transform = cv2.distanceTransform(predMask,cv2.DIST_L2,5)
    sure_fg = cv2.threshold(dist_transform,foreground_threshold*dist_transform.max(),255,0)
    sure_fg = cv2.dilate(sure_fg[1].astype(np.uint8), kernel3 ,iterations = 3)
    output = cv2.connectedComponentsWithStats(sure_fg, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)

    # Extract ConnectedComponent Output
    num_labels = output[0]
    markers = output[1]
    centroids = output[3]
 
    # Apply Watershed using Cell Markers and Label Cells
    labels = watershed(-dist_transform, markers, mask=predMask)
    labels = np.array(labels)

    return centroids[1:], labels

def getGT(frameID):
    mask = io.loadmat('Training Data 1D/Masks/mask' + str(frameID))
    mask = mask['LAB']
    mask = centreCrop(mask, 1024)
    output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
    centroids = output[3]
    markers = output[1]
    markersprint = (markers / markers.max() * 255).astype('uint8')
    imageio.imwrite('Test/' + str(frameID) + 'Labels.png', markersprint)

    return centroids[1:]

def centreCrop(image, new_size = 1024):
    h,w = image.shape[-2:]
    if len(image.shape) > 2:
        cropped_image = image[:, :, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
    else:
        cropped_image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
    return cropped_image

def getAccuracy(predCent, trueCent):
    centroidDiff = cdist(predCent, trueCent, 'euclidean')
    firstLabels, secondLabels = linear_sum_assignment(centroidDiff)
    #pdb.set_trace()
    accurateSegs = 0
    trueSeg = len(trueCent)

    for pred, true  in zip(firstLabels,secondLabels):
        if centroidDiff[pred, true] < 5 :
            accurateSegs += 1
            
    return accurateSegs / trueSeg

def inferNetworkBatch(images, num_images, net):

    ## Inference 
    outputs = [None] * num_images
    for idx, image in enumerate(images):
        with torch.no_grad():
            outputs[idx] = net(image)

    return outputs

def cellLabelLoss(x0, tl, net):

    runningAcc = 0
    predictions = inferNetworkBatch(images = tl.tensorsBW, num_images = tl.num_images, net = net)
    tl.makeMasks(predictions)

    for idx, predMask in enumerate(tl.masks):
        centroids, labels = label_cells(predMask, x0)
        accuracy = getAccuracy(centroids, getGT(idx))
        runningAcc += accuracy
        print(idx, x0, accuracy)

        labelprint = (labels / labels.max() * 255).astype('uint8')
        imageio.imwrite('Test/' + str(idx) + 'Labels.png', labelprint)

    if runningAcc == 0:
        return float('inf')
    else:
        return 1 / (runningAcc / tl.num_images)



def main():
    ## Make 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tl = Timelapse(device = device, image_dir = 'Training Data 1D/Images/')
    # Load image for inference 
    tl.loadImages(normalize = True, dimensions = 1024, toCrop = True)
    # Pass Image to Inference script, return predicted Mask

    ## Instantiate Net, load parameters
    net = Net()
    net.eval()
    checkpoint = torch.load("Current Model/model_cp.pt")
    net.load_state_dict(checkpoint['network'])
    net.to(device)

    print('starting')
    res = optimize.minimize_scalar(cellLabelLoss, args = (tl, net), bounds=(0, 1), method='bounded')
    print(res)

main()





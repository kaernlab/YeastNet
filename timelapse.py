import imageio as imio
import numpy as np
import torch
import torchvision as tv
import os
import pdb
import scipy.spatial.distance as scipyD
import scipy.optimize as scipyO
import matplotlib.pyplot as plt

import PIL
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from Utils.helpers import centreCrop

class Timelapse():
    def __init__(self, device, image_dir = "inference"):
        self.device = device
        self.toTensor = tv.transforms.ToTensor() 
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f)) and f[-3:] != 'ini']
        self.num_images = len(self.image_filenames)
        self.total_cells = 0

        self.tensorsBW = [None] * self.num_images
        self.imagesBW = [None] * self.num_images
        self.masks = [None] * self.num_images
        self.labels = [None] * self.num_images
        self.centroids = [None] * self.num_images
        self.identity = [None] * self.num_images
        self.contouredImages = [None] * self.num_images
        self.areas = [None] * self.num_images

        self.centreCrop = centreCrop

    def __getitem__(self,idx):
        x, gfpfl = self.BuildCellTrack(idx, 'GFP')
        x, rfpfl = self.BuildCellTrack(idx, 'RFP')
        return x, gfpfl, rfpfl

    def loadImages(self, dimensions = 1024, normalize = False, toCrop = True):
        """ Load all BrightField images.
        
        This method loops over the list of all file names in the image directory
        and loads the image. Each image is normalized, cropped, and formatted to
        fit the right dimensionality for network inference. The pytorch tensor 
        and uint8 version are stored in a list property on the object.

        Input:
            dimensions: width and length of desired cropped image size (1024px default)
            normalize: boolean input offering option to normalize image (False default)
            toCrop: boolean input offering option to crop image

        Output:
            None, computed values are stored on object
        """
        for idx, image_name in enumerate(self.image_filenames):
            path = self.image_dir + image_name
            imageBW = imio.imread(path) 

            if normalize:
                imageBW = self.normalizeGrayscale(imageBW)

            tensorBW = imageBW[:,:,np.newaxis].astype(np.double)
            tensorBW = self.toTensor(tensorBW).unsqueeze_(0).float()
            tensorBW = tensorBW.to(self.device)

            if toCrop == True:
                tensorBW = self.centreCrop(tensorBW, dimensions)
                imageBW = self.centreCrop(imageBW, dimensions) 

            self.tensorsBW[idx] = tensorBW
            self.imagesBW[idx] = (imageBW / imageBW.max() * 255).astype('uint8')
            

    def makeMasks(self, masks):
        """ Generate masks from network segmentation predictions.
        
        This method loops over the list of all mask predictions. Predictions are
        output from the network in a 3d array. They are moved back to the RAM from
        the gpu vram, and converted to a numpy array from a pytorch tensor. The
        array is separated into two mask (background and cell). The cell mask is 
        the desired mask. It is converted into uint8 and stored on the object.

        Input:
            masks: network segmentation prediction mask, dtype: cuda pytorch tensor

        Output:
            None, computed values are stored on object
        """
        for idx, mask in enumerate(masks):
            bgMask = mask.cpu().detach().numpy()[0,0,:,:]
            cellMask = mask.cpu().detach().numpy()[0,1,:,:]
            mask = np.zeros(bgMask.shape)
            mask = (cellMask > bgMask) * 1
            mask = mask / mask.max() * 255
            self.masks[idx] = mask.astype('uint8')

    def normalizeGrayscale(self, image):
        """ Normalizes grayscale images.

        Normalizes image by zero-centering the data and changing the standard deviation
        to 1. Then rescaling the data between 0-1 

        Input:
            image: image to be normalized
        Outputs:
            normalizedImage: normalized image
        """
        image = (image - image.mean()) / image.std()
        normalizedImage = (image - image.min()) / image.max()
        return normalizedImage

    def cellTrack(self):
        """ Tracks cells through segmented timelapse microscopy images.

        Detected Cells in the first image are numbered. These numbers are their identity. 
        Tracking occurs one frame at a time. The distance between the centroids of every
        cell in one frame and every cell in the subesequent frame is calculated and stru-
        ctured into a difference matrix. The row index indicates the cell identities in
        the initial frame and the column index indicates the cell labels (not identities)
        fromt he subsequent frame. The hungarian algorithm matches centroids most likely 
        to belong to the same cell. And identity array is stored indicating the identity of
        every cell in a frame.

        """
        self.identity[0] = np.unique(self.labels[0])[1:]
        self.total_cells = len(self.identity[0])

        for idx, (firstC, secondC, firstA, secondA) in enumerate(zip(self.centroids[:-1], self.centroids[1:], self.areas[:-1], self.areas[1:])):
            timepoint = idx+1
            centroidDiff = scipyD.cdist(firstC, secondC, 'euclidean')
            firstA = np.repeat(np.array(firstA)[:, np.newaxis], len(secondA), axis = 1)
            secondA = np.repeat(np.array(secondA)[np.newaxis, :], len(firstA), axis = 0)
            areaDiff = np.abs(firstA.astype('int32') - secondA.astype('int32'))
            Y = centroidDiff#     + areaDiff
            
            #pdb.set_trace()
            
            firstLabels, secondLabels = scipyO.linear_sum_assignment(Y)
            firstLabels = self.identity[timepoint-1]
            self.identity[timepoint] = np.full(len(secondC), -1)

            for idx, label in enumerate(secondLabels):
                self.identity[timepoint][label] = firstLabels[idx]

            for idx, label in enumerate(self.identity[timepoint]):
                if label == -1:
                    self.total_cells += 1
                    self.identity[timepoint][idx] = self.total_cells
                    
            
    def DrawTrackedCells(self):
        font_fname = 'Utils/Fonts/Roboto-Regular.ttf'
        font_size = 20
        font = ImageFont.truetype(font_fname, font_size)
    
        for imageID in range(self.num_images):
            #bw_image = ((self.imagesBW[imageID]/ self.imagesBW[imageID].max())*255)
            #bw_image = Image.fromarray(bw_image.astype('uint8')).convert('RGB')
            bw_image = Image.fromarray(self.contouredImages[imageID])#.convert('RGB')
            draw = ImageDraw.Draw(bw_image)

            for idx, (label, centroid) in enumerate(zip(self.identity[imageID], self.centroids[imageID])):
                draw.text((centroid[0]-5, centroid[1]-10), str(label), font=font, fill='rgb(255, 0, 0)')

            bw_image.save(self.image_dir + 'Results/Tracking/' + str(imageID) + 'Tracked.png')

        #os.system("ffmpeg -r 5 -i ./inference/Results/Tracked/%01dTracked.png -vcodec mpeg4 -y movie.mp4")

    def BuildCellTrack(self, idx, fp):

        outputfl = []
        x = []
        
        for timepoint in range(self.num_images):
            path = self.image_dir + fp + '/z1_t_000_000_%03d_'  % (timepoint+1) + fp + '.tif'
            imageGFP = imio.imread(path) 
            imageGFP = self.centreCrop(imageGFP, 1024)
            #imageGFP = (imageGFP / imageGFP.max() * 255).astype('uint8')
            
            trackedLabel = np.where((self.identity[timepoint]==idx))[0]
            if trackedLabel:
                trackedLabel = trackedLabel[0]
                area = self.areas[timepoint][trackedLabel]
                fl = imageGFP[self.labels[timepoint]==trackedLabel].sum() / area
                outputfl.append(fl)
                x.append(timepoint)
        
        return x, outputfl
        
        
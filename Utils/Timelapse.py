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
from Utils.helpers import autoCrop

class Timelapse():
    def __init__(self, device, image_dir):
        self.device = device
        self.toTensor = tv.transforms.ToTensor() 
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f)) and f[-3:] != 'ini']
        self.image_filenames.sort()
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
        self.setMoments = self.getDataSetMoments()

        self.centreCrop = centreCrop
        self.autoCrop = autoCrop

    def __getitem__(self,idx):
        path= self.image_dir[:-4]
        prefix='z2_t_000_000_'
        suffix='_'
        x, gfpfl, _ = self.BuildCellTrack(idx, fp='GFP', path = path, prefix=prefix, suffix=suffix)
        x, rfpfl, cell_area = self.BuildCellTrack(idx, fp='RFP', path = path, prefix=prefix, suffix=suffix)
        return x, gfpfl, rfpfl

    def loadImages(self, force_dimensions = 1024, normalize = False, toCrop = False):
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

            if len(imageBW.shape) > 2:
                #rgb to gray
                imageBW = imageBW[:,:,1]

            if normalize:
                imageBW = self.normalizeGrayscale(imageBW)

            tensorBW = imageBW[:,:,np.newaxis].astype(np.double)
            tensorBW = self.toTensor(tensorBW).unsqueeze_(0).float()
            #tensorBW = tensorBW.to(self.device)

            if toCrop == True:
                tensorBW = self.centreCrop(tensorBW, force_dimensions)
                imageBW = self.centreCrop(imageBW, force_dimensions)
            else:
                tensorBW = self.autoCrop(tensorBW)
                imageBW = self.autoCrop(imageBW)

            self.tensorsBW[idx] = tensorBW
            #self.imagesBW[idx] = (imageBW / imageBW.max() * 255).astype('uint8')
            self.imagesBW[idx] = imageBW#.astype('uint8')

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


    ##This function will load all images and get the mean and standard deviation pixel intensity
    def getDataSetMoments(self):
        setMoments = {}
        mean = np.array([])
        std = np.array([])
        for image_name in self.image_filenames:
            path = self.image_dir + image_name
            bw_image = imio.imread(path) 
            mean = np.append(mean, bw_image.mean())
            std = np.append(std, bw_image.std())

        setMoments['mean'] = mean.mean()
        setMoments['std'] = std.mean()

        return setMoments

    def normalizeGrayscale(self, image):
        """ Normalizes grayscale images.

        Normalizes image by zero-centering the data and converting the pixel
        values to z-scores. The mean and std used is collected for using every image in
        the dataset.

        Input:
            image: image to be normalized
        Outputs:
            normalizedImage: normalized image
        """
        image = (image - self.setMoments['mean']) / self.setMoments['std']
        image = image + abs(image.min())
        normalizedImage = image / image.max()
        #image = (image - image.min()) / image.max()
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
            bw_image = Image.fromarray((self.contouredImages[imageID]*255).astype('uint8'))#.convert('RGB')
            draw = ImageDraw.Draw(bw_image)

            for idx, (label, centroid) in enumerate(zip(self.identity[imageID], self.centroids[imageID])):
                draw.text((centroid[0]-5, centroid[1]-10), str(label), font=font, fill='rgb(255, 0, 0)')

            bw_image.save(self.image_dir + 'Results/Tracking/' + str(imageID) + 'Tracked.png')

        #os.system("ffmpeg -r 5 -i ./inference/Results/Tracked/%01dTracked.png -vcodec mpeg4 -y movie.mp4")

    def BuildCellTrack(self, idx, fp='GFP', path = '', prefix = '', suffix = ''):

        outputfl = []
        x = []
        cell_area = []
        
        for timepoint in range(self.num_images):
            image_path = path  + '/' +  fp + '/' + prefix + '%03d'  % (timepoint+1) + suffix + fp +'.tif'
            imageFP = imio.imread(image_path)
            imageFP = self.centreCrop(imageFP, 1024)
            #imageFP = (imageFP / imageFP.max() * 255).astype('uint8')
            
            trackedLabel = np.where((self.identity[timepoint]==idx))[0]
            #pdb.set_trace()
            if trackedLabel.size==1:
                trackedLabel = trackedLabel[0]
                area = self.areas[timepoint][trackedLabel]
                fl = imageFP[self.labels[timepoint]==(trackedLabel+1)].sum() / area
                outputfl.append(fl)
                cell_area.append(area)
                x.append(timepoint)
        
        return x, outputfl, cell_area
    
    def GetFlData(self, timepoint, fp='GFP', path = '', prefix = '', suffix = ''):
        """ 
        This method returns arrays containing all the data for each cell detected
        at a timepoint. Lowest timepoint input is assumed to be 1 (therefore
        subtracted by one for indexing)."""

        outputfl = []
        cell_area = []
        timepointID = timepoint-1

        image_path = path  + '/' +  fp + '/' + prefix + '%03d'  % (timepoint+1) + suffix + '.tif'
        imageFP = imio.imread(image_path) 
        imageFP = self.centreCrop(imageFP, 1024)
        for idx, cellID in enumerate(self.identity[timepointID]):
            area = self.areas[timepointID][idx]
            fl = imageFP[self.labels[timepointID]==idx+1].sum() / area
            outputfl.append(fl)
            cell_area.append(area)

        return outputfl, cell_area, self.identity[timepointID]

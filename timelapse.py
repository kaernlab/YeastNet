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


class Timelapse():
    def __init__(self, device, image_dir = "inference"):
        self.device = device
        self.toTensor = tv.transforms.ToTensor()
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(self.image_dir + '/BW') if os.path.isfile(os.path.join(self.image_dir + '/BW', f))]
        self.num_images = len(self.image_filenames)
        self.total_cells = 0

        self.tensorsBW = [None] * self.num_images
        self.imagesBW = [None] * self.num_images#np.array([self.num_images])
        self.masks = [None] * self.num_images#np.array([self.num_images])
        self.labels = [None] * self.num_images#np.array([self.num_images])
        self.centroids = [None] * self.num_images#np.array([self.num_images])
        self.identity = [None] * self.num_images
        self.contouredImages = [None] * self.num_images
        self.areas = [None] * self.num_images

    def loadImages(self, dimensions = 1024, normalize = False):
        for idx, image_name in enumerate(self.image_filenames):
            path = self.image_dir + '/BW/' + image_name
            imageBW = imio.imread(path) 

            if normalize:
                imageBW = self.normalizeGrayscale(imageBW, imageBW.mean(), imageBW.std())

            tensorBW = imageBW[:,:,np.newaxis].astype(np.double)
            tensorBW = self.toTensor(tensorBW).unsqueeze_(0).float()
            tensorBW = self.centreCrop(tensorBW.to(self.device), dimensions)

            self.tensorsBW[idx] = tensorBW
            imageBW = self.centreCrop(imageBW, dimensions) 
            self.imagesBW[idx] = (imageBW / imageBW.max() * 255).astype('uint8')
            

    def makeMasks(self, masks):
        for idx, mask in enumerate(masks):
            bgMask = mask.cpu().detach().numpy()[0,0,:,:]
            cellMask = mask.cpu().detach().numpy()[0,1,:,:]
            mask = np.zeros(bgMask.shape)
            mask = (cellMask > bgMask) * 1
            mask = mask / mask.max() * 255
            self.masks[idx] = mask.astype('uint8')

    def centreCrop(self, image, new_size):
        h,w = image.shape[-2:]
        if len(image.shape) > 2:
            image = image[:, :, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        else:
            image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
        return image

    def normalizeGrayscale(self, image, mean, std):
        image = (image - mean) / std
        image = (image - image.min()) / image.max()
        return image

    def cellTrack(self):
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

            bw_image.save('inference/Results/Tracked/' + str(imageID) + 'Trackedg.png')

        os.system("ffmpeg -r 5 -i ./inference/Results/Tracked/%01dTracked.png -vcodec mpeg4 -y movie.mp4")

    def BuildCellTrack(self, idx, fp):

        outputfl = []
        x = []
        
        for timepoint in range(30):
            path = self.image_dir + '/' + fp + '/z1_t_000_000_%03d_'  % (timepoint+1) + fp + '.tif'
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
        
        
import csv
import cv2
import os
import pdb
from scipy import io
from Utils.helpers import centreCrop


def testMeasureF(tl, singleFile = True, makeTG = False, platform = 'YeastNet'): #, testPrediction = False, makeTG = False

    if singleFile:
        with open(tl.image_dir + 'Results/' + platform + '/yn_seg_and_track.csv', 'w', newline='') as csvfile:
            fieldnames = ['Frame_Number','Cell_number', 'Cell_colour', 'Position_X', 'Position_Y', 'Unique_cell_number']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for frameID in range(tl.num_images):
                predCent = tl.centroids[frameID]
                for cellNumber, (cellID, pred) in enumerate(zip(tl.identity[frameID], predCent)):
                    writer.writerow({
                        'Frame_Number': frameID+1,
                        'Cell_number': cellNumber+1,
                        'Cell_colour': 0,
                        'Position_X': pred[0],
                        'Position_Y': pred[1],
                        'Unique_cell_number': cellID
                        })

    else:
        ## Rarely if ever used, *not updated for tracking yet*

        for frameID in range(tl.num_images):

            predCent = tl.centroids[frameID]

            with open(tl.image_dir + 'Results/YeastNet/yn_seg_and_track' + str(frameID) + '.csv', 'w', newline='') as csvfile:
                fieldnames = ['Cell_number', 'Cell_colour', 'Position_X', 'Position_Y']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for cellID, pred in enumerate(predCent):
                    writer.writerow({
                        'Cell_number': cellID+1,
                        'Cell_colour': 0,
                        'Position_X': pred[0],
                        'Position_Y': pred[1]})


    if makeTG == True:
        ## Makes GroundTruth csv file for Seg/Tracking Accuracy measurement. 


        if not os.path.isdir(tl.image_dir + 'Results/GroundTruth'):
            os.mkdir(tl.image_dir + 'Results/GroundTruth')

        with open(tl.image_dir + 'Results/GroundTruth/gt_seg_and_track.csv', 'w', newline='') as csvfile:

            fieldnames = ['Frame_Number','Cell_number', 'Cell_colour', 'Position_X', 'Position_Y','Unique_cell_number']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
            writer.writeheader()
            for frameID in range(tl.num_images):

                mask = io.loadmat('Training Data 1D/Masks/mask' + str(frameID) + '.mat')
                mask = centreCrop(mask['LAB'], 1024)
                output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
                centroids = output[3]
                trueCent = centroids[1:]
                

                predCent = tl.centroids[frameID]
                for cellID, true in enumerate(trueCent):
                    writer.writerow({
                        'Frame_Number': frameID+1,
                        'Cell_number': cellID+1,
                        'Cell_colour': 0,
                        'Position_X': true[0],
                        'Position_Y': true[1],
                        'Unique_cell_number': int(mask[[int(true[1])],[int(true[0])]])
                        })

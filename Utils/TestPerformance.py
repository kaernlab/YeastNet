import csv
import cv2
import os
import pdb
import torch
import scipy.io as sio
import numpy as np
from Utils.helpers import centreCrop
from Utils.helpers import accuracy


def makeResultsCSV(tl, model_num = 0, singleFile = True, makeTG = False, platform = 'YeastNet'): #, testPrediction = False, makeTG = False

    if singleFile:
        with open(tl.image_dir + 'Results/yn_seg_and_track.csv', 'w', newline='') as csvfile:
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

            with open(tl.image_dir + 'Results/yn_seg_and_track' + str(frameID) + '.csv', 'w', newline='') as csvfile:
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

        #model = torch.load('./CrossValidation/Finetuned Models/model_cp' + str(model_num) + '.pt')
        #testIDs = model['testID']
        #testIDs.sort()
        testIDs = list(range(50))
        with open(tl.image_dir + 'Results/gt_seg_and_track.csv', 'w', newline='') as csvfile:

            fieldnames = ['Frame_Number','Cell_number', 'Cell_colour', 'Position_X', 'Position_Y','Unique_cell_number']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
            writer.writeheader()



            for idx, testID in enumerate(testIDs):

                mask = sio.loadmat('Training Data 1D/Masks/mask' + str(testID) + '.mat')
                mask = centreCrop(mask['LAB'], 1024)
                output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
                centroids = output[3]
                trueCent = centroids[1:]
                

                predCent = tl.centroids[idx]
                for cellID, true in enumerate(trueCent):
                    writer.writerow({
                        'Frame_Number': idx+1,
                        'Cell_number': cellID+1,
                        'Cell_colour': 0,
                        'Position_X': true[0],
                        'Position_Y': true[1],
                        'Unique_cell_number': int(mask[[int(true[1])],[int(true[0])]])
                        })

def makeAccCSV(model_path, image_dir):

    checkpoint = torch.load(model_path)
    testIDs = checkpoint['testID']
    testIDs.sort()

    with open(image_dir + 'Results/old_seg_and_track.csv', 'w', newline='') as csvfile:
        fieldnames = ['Frame_Number','Cell_number', 'Cell_colour', 'Position_X', 'Position_Y', 'Unique_cell_number']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for frameID, testID in enumerate(testIDs):

            if testID < 51:
                pred_mask_name = 't_%.3d.mat' % (testID+1)
            elif testID > 101:
                pred_mask_name = 't_%.3d.mat' % (testID-101)
            else:
                pred_mask_name = 't_%.3d.mat' % (testID-50)

            mask = sio.loadmat('Old Method/' + pred_mask_name)
            mask = mask['LAB_orig']
            mask = centreCrop(mask, 1024)
            pred_mask = (mask != 0)*1
            output = cv2.connectedComponentsWithStats(pred_mask.astype(np.uint8), 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
            centroids = output[3]
            predCent = centroids[1:]

            identity = []
            for idx, centroid in enumerate(centroids):
                x = centroid.astype(int)
                identity.append(mask[x[1],x[0]])

            #pdb.set_trace()
            for cellNumber, (cellID, pred) in enumerate(zip(identity,predCent)):
                writer.writerow({
                    'Frame_Number': frameID+1,
                    'Cell_number': cellNumber+1,
                    'Cell_colour': 0,
                    'Position_X': pred[0],
                    'Position_Y': pred[1],
                    'Unique_cell_number': cellID
                    })


def compareOld2(model_path):

    runningIOU = 0
    runningPA = 0
    checkpoint = torch.load(model_path)
    testIDs = checkpoint['testID']
    testIDs.sort()

    for testID in testIDs:
        true_mask = sio.loadmat('Training Data 1D/Masks/mask' + str(testID) + '.mat')
        true_mask = (true_mask['LAB_orig'] != 0)*1
        true_mask = centreCrop(true_mask, 1024)

        if testID < 51:
            pred_mask_name = 't_%.3d.mat' % (testID+1)
        elif testID > 101:
            pred_mask_name = 't_%.3d.mat' % (testID-101)
        else:
            pred_mask_name = 't_%.3d.mat' % (testID-50)

        pred_mask = sio.loadmat('Old Method/' + pred_mask_name)
        pred_mask = (pred_mask['LAB_orig'] != 0)*1
        pred_mask = centreCrop(pred_mask, 1024)

        PixAccuracy, IntOfUnion = accuracy(true_mask, pred_mask)
        #pdb.set_trace()
        runningIOU += IntOfUnion[1]

    return runningIOU / 15
import numpy as np
import os
import pdb
import torch
import scipy.io as sio
import cv2
import csv

## Import Custom Code
from Utils.helpers import accuracy
from Utils.helpers import centreCrop


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
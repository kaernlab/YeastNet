
## PURPOSE
# The Purpose of this function is to quantify the performance of
# the CellStar Platform when used on my GT-annotated dataset. The 
# script will generate an IOU measure, and a segmentation accuracy
# which will be reported against the same measures from YeastNet.
# 

import scipy.io as sio
import numpy as np
import cv2
import csv 
import pdb
import torch
from Utils.helpers import accuracy
from Utils.helpers import centreCrop



model = 3
kernel = np.ones((2,2), np.uint8)
pred_path = 'C:/Users/Danny/Desktop/yeast-net/CrossValidation/CrossVal Accuracy/Model' + str(model) + '/CellStar/segments/'
runningIoU = 0

cp = torch.load('C:/Users/Danny/Desktop/yeast-net/CrossValidation/Finetuned Models/model_cp' + str(model) + '.pt')
testIDs = cp['testID']

for idx in testIDs:
    if idx < 51:
        pred_mask_name = 'z1_t_000_000_%03d_BF_segmentation.mat' % (idx+1)
    elif idx > 101:
        pred_mask_name = 'z3_t_000_000_%03d_BF_segmentation.mat' % (idx-101)
    else:
        pred_mask_name = 'z2_t_000_000_%03d_BF_segmentation.mat' % (idx-50)

    pred_mask =  sio.loadmat(pred_path + pred_mask_name)
    pred_mask = (pred_mask['segments'] != 0)*1

    true_mask = sio.loadmat('Training Data 1D/Masks/mask' + str(idx) + '.mat')
    true_mask = (true_mask['LAB_orig'] != 0)*1

    #pred_mask = centreCrop(pred_mask, 1024)
    true_mask = centreCrop(true_mask, 1024)
    PixAccuracy, IntOfUnion = accuracy(true_mask, pred_mask)

    runningIoU += IntOfUnion[1]

    #pdb.set_trace()

print(runningIoU / 15)



'''
with open(pred_path + '/yn_seg.csv', 'w', newline='') as csvfile:
    fieldnames = ['Frame_Number','Cell_number', 'Cell_colour', 'Position_X', 'Position_Y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for frameID in range(1,52):

        pred_mask_name = 'z2_t_000_000_%03d_BF_segmentation.mat' % frameID
        pred_mask =  sio.loadmat(pred_path + pred_mask_name)
        pred_mask = (pred_mask['segments'] != 0)*1
        pred_mask = cv2.erode(pred_mask.astype(np.uint8), kernel, iterations = 3)
        #pdb.set_trace()
        output = cv2.connectedComponentsWithStats(pred_mask, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
        centroids = output[3]
        predCent = centroids[1:]

        for cellID, pred in enumerate(predCent):
            writer.writerow({
                'Frame_Number': frameID,
                'Cell_number': cellID+1,
                'Cell_colour': 0,
                'Position_X': pred[0],
                'Position_Y': pred[1]})
'''
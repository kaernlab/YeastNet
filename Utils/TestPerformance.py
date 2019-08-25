import csv
import cv2
import os
import pdb
import torch
import scipy.io as sio
import numpy as np
import imageio
import matplotlib.pyplot as plt
from Utils.helpers import centreCrop
from Utils.helpers import accuracy


def makeResultsCSV(tl, model_num = 0, singleFile = True, makeTG = False, platform = 'YeastNet'): #, testPrediction = False, makeTG = False
    ''' Make csv with seg/tracking results
    
    Given a model path and an image directory, the segmentation 
    and tracking results are compiled into a csv file that
    can be used to benchmark performance using the YIT-Benchmark. THe same
    comparison made in the CEllStar paper. It can also make a ground truth csv file
    as required by YIT-Benchmark'''

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
    ''' Make csv with seg/tracking results of Old Method
    
    Given a model path and an image directory, the segmentation 
    and tracking results of our Old Method, heavily modified version
    of the Doncic et al. (2013) method are compiled into a csv file that
    can be used to benchmark performance using the YIT-Benchmark. THe same
    comparison made in the CEllStar paper.'''

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

        _, IntOfUnion = accuracy(true_mask, pred_mask)
        #pdb.set_trace()
        runningIOU += IntOfUnion[1]

    return (runningIOU / len(testIDs))


def cellStarIOU(modelnum = 10, dataset = 'DSDataset'):
    """ Produce IOU measure for CellStar Output
    
    The Purpose of this function is to quantify the performance of
    the CellStar Platform when used on my GT-annotated dataset. The 
    script will generate an IOU measure"""

    #pred_path = 'C:/Users/Danny/Desktop/yeast-net/CrossValidation/CrossVal Accuracy/Model' + str(modelnum) + '/CellStar/segments/'
    pred_path = 'G:/CellStar/Images/Datasets/' + dataset + '/segments/'
    runningIOU = 0

    #cp = torch.load('./NewNormModels/new_norm_testV3K%01d.pt' % modelnum)
    cp = torch.load('./CrossValidation/DoubleTrainedModels/model_cp%01d.pt' % modelnum)
    temp = cp['testID']
    testIDs = temp.pop(dataset, None)

    for idx in testIDs:
        pred_mask_name = 'im%03d_segmentation.mat' % (idx+1)

        pred_mask =  sio.loadmat(pred_path + pred_mask_name)
        pred_mask = (pred_mask['segments'] != 0)*1

        true_mask = np.load('./Datasets/' + dataset + '/Masks/mask%03d.npy' % idx)
        true_mask = (true_mask != 0)*1

        _, IntOfUnion = accuracy(true_mask, pred_mask)
        '''
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(true_mask) 
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask)
        plt.show()
'''
        runningIOU += IntOfUnion[1]


    return (runningIOU / len(testIDs))

def yeastSpotterIOU(data_folder="./"):
    """ Produce IOU measure for YeastSpotter Output
    
    The Purpose of this function is to quantify the performance of
    the YeastSpotter Platform. The script will generate an IOU measure"""

    data_folder = '../yeast-net-backup/Datasets Backup/YITDataset1/'
    mask_folder = data_folder + 'MasksOLD/'
    results_folder = data_folder + 'RCresults/'
    filetype = '.tif'
    runningIOU = 0
    testIDs = list(range(60))

    for testID in testIDs:

        
        #pred_mask = imageio.imread(results_folder + ('im%03d' % testID) + filetype)
        #pred_mask = (pred_mask != 0)*1

        pred_mask = sio.loadmat(results_folder + 't_%03d_z1_BW' % testID)
        pred_mask = (pred_mask['BW'] != 0)*1
 

        true_mask = imageio.imread(mask_folder + "mask%03d" % testID + filetype)
        true_mask = (true_mask != 0)*1

        _, IntOfUnion = accuracy(true_mask, pred_mask)

        runningIOU += IntOfUnion[1]


    return (runningIOU / len(testIDs))

def getMeanAccuracy(tl, model_path=None, data_folder='./'):
    ''' Generate a Mean IOU for a YeastNet Model
    
    This method accepts a model checkpoint and a timelapse
    object and loops over the test set associated with the
    model to get a set of cell IOU measurements. It then outputs
    a mean cell IOU.'''
    
    runningIOU = 0
    if model_path:
        checkpoint = torch.load(model_path)
        testIDs = checkpoint['testID']
        testIDs.sort()
    else:
        testIDs = list(range(tl.num_images))

    mask_folder = data_folder + "/Masks/"
    
    mask_filenames = [f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f)) and f[-3:] != 'ini']
    filetype = mask_filenames[0][-4:]
    #pdb.set_trace()
    for testID, pred_mask in zip(testIDs, tl.masks):

        if filetype == '.mat':
            true_mask = sio.loadmat(mask_folder + "mask" + str(testID) + filetype)
            true_mask = (true_mask['LAB_orig'] != 0)*1
            true_mask = centreCrop(true_mask, 1024)
        else:
            true_mask = imageio.imread(mask_folder + "mask" + str(testID) + filetype)
            true_mask = (true_mask != 0)*1

        pred_mask = (pred_mask != 0)*1
        _, IntOfUnion = accuracy(true_mask, pred_mask)

        runningIOU += IntOfUnion[1]

    
    return (runningIOU / len(testIDs))


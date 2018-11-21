import csv
import cv2
import imageio
from scipy import io
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def testMeasureF(tl, testPrediction = False):
    runningAcc = 0
    for frameID in range(tl.num_images):
        mask = io.loadmat(tl.image_dir + '/Masks/mask' + str(frameID))
        mask = mask['LAB']
        mask = tl.centreCrop(mask, 1024)

        output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)#, cv2.CCL_DEFAULT)
        markers = output[1]
        centroids = output[3]
        imageio.imwrite(tl.image_dir + '/Results/' + str(frameID) + 'LabelsT.png', markers)

        predCent = tl.centroids[frameID]
        trueCent = centroids[1:]

        with open(tl.image_dir + '/TestSet/YeastNet/yn_seg' + str(frameID) + '.csv', 'w', newline='') as csvfile:
            fieldnames = ['Cell_number', 'Cell_colour', 'Position_X', 'Position_Y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for cellID, pred in enumerate(predCent):
                writer.writerow({
                    'Cell_number': cellID+1,
                    'Cell_colour': 0,
                    'Position_X': pred[0],
                    'Position_Y': pred[1]})
        
        
        with open(tl.image_dir + '/TestSet/GroundTruth/gt_seg' + str(frameID) + '.csv', 'w', newline='') as csvfile:
            fieldnames = ['Cell_number', 'Cell_colour', 'Position_X', 'Position_Y']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for cellID, pred in enumerate(trueCent):
                writer.writerow({
                    'Cell_number': cellID+1,
                    'Cell_colour': 0,
                    'Position_X': pred[0],
                    'Position_Y': pred[1]})
        
        ## Testing Generally done using Evaluation Platform, This is no longer needed
        if testPrediction == True:
            centroidDiff = cdist(predCent, trueCent, 'euclidean')
            firstLabels, secondLabels = linear_sum_assignment(centroidDiff)

            accurateSeg = 0
            trueSeg = len(trueCent)


            for pred, true  in zip(firstLabels,secondLabels):
                
                print(centroidDiff[pred, true])
                if centroidDiff[pred, true] < 5 :
                    accurateSeg += 1
                    
            print(accurateSeg, trueSeg)
            runningAcc += accurateSeg / trueSeg

            print(runningAcc / tl.num_images)
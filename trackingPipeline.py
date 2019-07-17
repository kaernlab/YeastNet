import argparse
import pdb
from Utils.makeTimelapse import makeTimelapse
from Utils.TestPerformance import makeResultsCSV, getMeanAccuracy

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f","--datasetfolder", type=str, help="Input string of dataset folder name. Default is current Directory. Assumed directory structure is '/datasetfolder/Images/(set of images)'", default="./")
parser.add_argument("-p","--make_plot", type=bool, help="Boolean variable to choose whether fluorescence plots are desired", default=False)
parser.add_argument("-c","--make_csv", type=bool, help="Boolean variable to choose whether csv is made for Benchmarking", default=False)
parser.add_argument("-i","--compare_iou", type=bool, help="Boolean variable to choose whether IOU measurement is made", default=False)
args = parser.parse_args()

datasetfolder = args.datasetfolder
makePlots = args.make_plot
makeCSV = args.make_csv
compareIOU = args.compare_iou
model_path = './ynetmodel/trainedModel.pt'
model_path = './CrossValidation/DoubleTrainedModels/model_cp10.pt'
datasetfolder = 'TestSets/YITDataset1'
compareIOU = False

## For testing purposes
#imagedir = '../tracking-analysis-py/Data/stable_60/FOV4/BF/'


tl = makeTimelapse('./' + datasetfolder + '/Images/', model_path)

if compareIOU:
    print(getMeanAccuracy(tl, data_folder = datasetfolder))
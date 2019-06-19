import argparse
import pdb
from Utils.makeTimelapse import makeTimelapse
from Utils.TestPerformance import makeResultsCSV, getMeanAccuracy

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f","--imagefolder", type=str, help="Input string of image folder name. Default is current Directory", default="./")
parser.add_argument("-p","--make_plot", type=bool, help="Boolean variable to choose whether fluorescence plots are desired", default=False)
parser.add_argument("-c","--make_csv", type=bool, help="Boolean variable to choose whether csv is made for Benchmarking", default=False)
parser.add_argument("-i","--compare_iou", type=bool, help="Boolean variable to choose whether IOU measurement is made", default=False)
args = parser.parse_args()

imagefolder = args.imagefolder
makePlots = args.make_plot
makeCSV = args.make_csv
compareIOU = args.compare_iou
model_path = './ynetmodel/trainedModel.pt'

## For testing purposes
#imagedir = '../tracking-analysis-py/Data/stable_60/FOV4/BF/'


tl = makeTimelapse('./' + imagefolder + '/', model_path)

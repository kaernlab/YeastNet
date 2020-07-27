import argparse
import pdb
from Utils.makeTimelapse import makeTimelapse
from Utils.TestPerformance import makeResultsCSV, getMeanAccuracy
from Utils.plotFLTrack import plotFlTrack 
import pickle

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f","--datasetfolder", type=str, help="Input string of dataset folder name. Assumed directory structure is './datasetfolder/(set of images)'", default="BF")
parser.add_argument("-p","--make_plot", type=bool, help="Boolean variable to choose whether fluorescence plots are desired", default=False)
parser.add_argument("-s","--save_experiment", type=bool, help="Boolean variable to choose whether to save a file containing all TImelapse info", default=False)
args = parser.parse_args()

datasetfolder = args.datasetfolder
makePlots = args.make_plot
saveExp = args.save_experiment
model_path = './Published/YNModelParams.pt'


tl = makeTimelapse('./Images/' + datasetfolder + '/' , model_path, saveExp)
#tl = makeTimelapse('./Published/Images/Z2/', model_path, False)


if makePlots:
    plotFlTrack(tl)

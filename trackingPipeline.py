import argparse
import pdb
from Utils.makeTimelapse import makeTimelapse
from Utils.TestPerformance import makeResultsCSV, getMeanAccuracy

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f","--datasetfolder", type=str, help="Input string of dataset folder name. Default is current Directory. Assumed directory structure is '/datasetfolder/Images/(set of images)'", default="./")
parser.add_argument("-p","--make_plot", type=bool, help="Boolean variable to choose whether fluorescence plots are desired", default=False)
parser.add_argument("-s","--save_experiment", type=bool, help="Boolean variable to choose whether to save a file containing all TImelapse info", default=False)
args = parser.parse_args()

datasetfolder = args.datasetfolder
makePlots = args.make_plot
saveExp = args.save_experiment
model_path = './yeastnetparam.pt'


## For testing purposes
#imagedir = '../tracking-analysis-py/Data/stable_60/FOV4/BF/'


tl = makeTimelapse('./' + datasetfolder + '/Images/', model_path, saveExp)

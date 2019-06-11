import argparse
import pdb
from Utils.makeTimelapse import makeTimelapse

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imagefolder", type=str, help="Input string of image folder name. Default is current Directory", default="./")
parser.add_argument("--make_plot", type=str, help="Boolean variable to choose whether fluorescence plots are desired", default=False)
args = parser.parse_args()

imagefolder = args.imagefolder
makePlots = args.make_plot
model_path = './ynetmodel/trainedModel.pt'

## For testing purposes
#imagedir = '../tracking-analysis-py/Data/stable_60/FOV4/BF/'


tl = makeTimelapse('./' + imagefolder + '/', model_path)

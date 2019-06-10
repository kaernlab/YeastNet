import argparse
from Utils.makeTimelapse import makeTimelapse

## Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", type=str, help="Input string of image directory. Default is current Directory", default="./")
parser.add_argument("--model_num", type=str, help="Cross Validation model number to use", default="0")
parser.add_argument("--make_plot", type=str, help="Boolean variable to choose whether fluorescence plots are desired", default=False)
args = parser.parse_args()

imagedir = args.imagedir
model_num = args.model_num
makePlots = args.make_plot

## For testing purposes
imagedir = '../tracking-analysis-py/Data/stable_60/FOV4/BF/'
model_path = './TrackingTest/trackingtest.pt'


tl = makeTimelapse(imagedir, model_path)

## Import Librarys and Modules
import torch
import pdb
import matplotlib.pyplot as plt
import os
import imageio
import argparse

from ynetmodel.defineNetwork import Net
from Utils.helpers import getDatasetMoments
import ynetmodel.validateNetwork as validateNetwork
from ynetmodel.YeastSegmentationDataset import YeastSegmentationDataset

parser = argparse.ArgumentParser()
parser.add_argument("-k","--kfold", type=int, help="k-fold number", default=1)
parser.add_argument("-n","--normtype", type=int, help="Normalization type. Best to ignore and use default", default=3)
args = parser.parse_args()


def main(args = args):
    ## Instantiate Net, Load Options
    net = Net()
    k=args.kfold
    normtype = args.normtype

    # Load Model Parameters and set options
    #ImageIDs = torch.load("./NewNormModels/new_norm_testV3K%01d.pt" % k)
    #checkpoint = torch.load("./NewNormModels/new_norm_testV%01dK%01d.pt" % (normtype, k))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ImageIDs = torch.load('YeastNet2Param.pt', map_location=device)
    checkpoint = torch.load('YeastNet2Param.pt', map_location=device)
    net.to(device)
    net.load_state_dict(checkpoint['network'])
    net.eval()

    # Get Mean and STD for training set in each Dataset
    setMoments = getDatasetMoments(ImageIDs['trainID'])
    setMoments3 = setMoments
    setMoments1 = {'DSDataset': setMoments3.pop('DSDataset')}
    setMoments2 = {'YITDataset1': setMoments3.pop('YITDataset1')}

    # Get image testID for each dataset
    testIDs3 = ImageIDs['testID']
    testIDs1 = {'DSDataset': testIDs3.pop('DSDataset')}
    testIDs2 = {'YITDataset1': testIDs3.pop('YITDataset1')}

    testIDs = [testIDs1, testIDs2, testIDs3]
    setMoments = [setMoments1, setMoments2, setMoments3]


    testDataSets = [YeastSegmentationDataset(IDs, normtype = normtype, setMoments = setMoment) for (IDs, setMoment) in zip(testIDs,setMoments)]
    testLoaders = [torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=1) for dataset in testDataSets]
    val_acc = [validateNetwork.validate(net, device, loader, saveImages = True) for loader in testLoaders]


    print(val_acc)

if __name__ == '__main__':
    main()
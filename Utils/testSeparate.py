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
parser.add_argument("-n","--normtype", type=int, help="normtype", default=3)
args = parser.parse_args()


def main(args = args):
    ## Instantiate Net, Load Parameters, Move Net to GPU
    net = Net()
    k=args.kfold
    normtype = args.normtype
    ##Send Model to GPU
    checkpoint = torch.load("../NewNormModels/new_norm_testDSV%01dK%01d.pt" % (normtype, k))
    #checkpoint = torch.load("./TrackingTest/trackingtest.pt")
    #checkpoint = torch.load("./CrossValidation/Finetuned Models/model_cp%01d.pt" % k)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.load_state_dict(checkpoint['network'])
    net.eval()

    #checkpoint = torch.load("./NewNormModels/new_norm_testV2K%01d.pt" % k)
    checkpoint = torch.load("../CrossValidation/DoubleTrainedModels/model_cp%01d.pt" % k)

    trainIDs = checkpoint['trainID']
    setMoments = getDatasetMoments(trainIDs)
    setMoments3 = setMoments
    setMoments1 = {'DSDataset': setMoments3.pop('DSDataset')}
    setMoments2 = {'YITDataset1': setMoments3.pop('YITDataset1')}

    testIDs3 = checkpoint['testID']
    testIDs1 = {'DSDataset': testIDs3.pop('DSDataset')}
    testIDs2 = {'YITDataset1': testIDs3.pop('YITDataset1')}
    #pdb.set_trace()

    testIDs = [testIDs1, testIDs2, testIDs3]
    setMoments = [setMoments1, setMoments2, setMoments3]

    testDataSets = [YeastSegmentationDataset(IDs, normtype = normtype, setMoments = setMoment) for (IDs, setMoment) in zip(testIDs,setMoments)]
    testLoaders = [torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=1) for dataset in testDataSets]
    val_acc = [validateNetwork.validate(net, device, loader, saveImages = True) for loader in testLoaders]

    #testDataSetDS = YeastSegmentationDataset(testIDs1)
    #testDataSetY1 = YeastSegmentationDataset(testIDs2)
    #testDataSetY3 = YeastSegmentationDataset(testIDs3)
    #testLoader1 = torch.utils.data.DataLoader(testDataSetDS, batch_size=1,shuffle=False, num_workers=1)
    #testLoader2 = torch.utils.data.DataLoader(testDataSetY1, batch_size=1,shuffle=False, num_workers=1)
    #testLoader3 = torch.utils.data.DataLoader(testDataSetY3, batch_size=1,shuffle=False, num_workers=1)
    #val_acc1 = validateNetwork.validate(net, device, testLoader1, saveImages = True)
    #val_acc2 = validateNetwork.validate(net, device, testLoader2, saveImages = True)
    #val_acc3 = validateNetwork.validate(net, device, testLoader3, saveImages = True)

    print(val_acc)

    '''
    bw_image = [loadimage('./Datasets/DSDataset/Images/im%03d.tif' % idx) for idx in testIDs['DSDataset']]

    for idx in testIDs['YITDataset1']:
        bw_image = imageio.imread('./Datasets/DSDataset/Images/im%03d.tif' % idx)

    for idx in testIDs['YITDataset3']:
        bw_image = imageio.imread('./Datasets/DSDataset/Images/im%03d.tif' % idx)


    def loadimage(filename):

        imageio.imread(filename)
        image = (image - self.setMoments[dataset_name]['mean']) / self.setMoments[dataset_name]['std']
        #image = (image - image.mean())
        #image = (image / image.std())
        image = image + abs(image.min())
        #image = image - abs(image.min())
        image = image / image.max()
        image[:,:,numpy.newaxis].astype(numpy.double)

        return image'''

if __name__ == '__main__':
    main()
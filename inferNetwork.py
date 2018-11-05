import torch

import processImages as pi
from defineNetwork import Net
from processImages import YeastSegmentationDataset


def inferNetworkBatch():
    ## Make Inference DataLoader
    samplingList = list(range(pi.num_images()))
    yeastDataset = YeastSegmentationDataset(samplingList, crop_size = 1024)
    inferenceLoader = torch.utils.data.DataLoader(yeastDataset, batch_size=1,
                                            shuffle=False, num_workers=0)

    ## Instantiate Net, load parameters
    net = Net()
    net.eval()
    net.load_state_dict(torch.load("100epochmodel.pt"))
    ## Move Net to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    ## Inference 
    #for 
    with torch.no_grad():
        outputs = net(validationImage.float())


def inferNetworkSingle(image):
    ## Instantiate Net, load parameters
    net = Net()
    net.eval()
    net.load_state_dict(torch.load("./Current Model/model.pt"))
    ## Move Net to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    ## Inference 
    with torch.no_grad():
        outputs = net(image)

    return outputs
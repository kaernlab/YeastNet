import torch
import pdb

from defineNetwork import Net


def inferNetworkBatch(images, num_images, device = "cpu"):
    ## Instantiate Net, load parameters
    net = Net()
    net.eval()
    checkpoint = torch.load("Current Model/model_cp.pt")
    net.load_state_dict(checkpoint['network'])

    ## Move Net to GPU
    net.to(device)
    ## Inference 

    outputs = [None] * num_images
    for idx, image in enumerate(images):
        image = image.to(device)

        with torch.no_grad():
            outputs[idx] = net(image)

        image = image.to(torch.device("cpu"))
        #pdb.set_trace()

    return outputs

def inferNetworkSingle(image, device = "cpu"):
    ## Instantiate Net, load parameters
    net = Net()
    net.eval()
    checkpoint = torch.load("Current Model/model_cp.pt")
    net.load_state_dict(checkpoint['network'])

    ## Move Net to GPU
    net.to(device)
    ## Inference 
    with torch.no_grad():
        outputs = net(image)

    return outputs
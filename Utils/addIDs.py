import torch
import random
import pdb

def addIDs():

    IDsYIT1 = list(range(60))
    IDsYIT3 = list(range(20))

    random.shuffle(IDsYIT1)
    random.shuffle(IDsYIT3)

    for i in range(10):
        x = torch.load('/media/danny/Projects/yeast-net/CrossValidation/Finetuned Models/model_cp%01d.pt' % (i+1))
        first1=0+i*6
        last1= 6+i*6
        first3=0+i*2
        last3= 2+i*2
        newTestIDs = {
            'DSDataset': x['testID'],
            'YITDataset1': IDsYIT1[first1:last1],
            'YITDataset3': IDsYIT3[first3:last3]
        }

        newTrainIDs = {
            'DSDataset': x['trainID'],
            'YITDataset1': [z for z in IDsYIT1 if z not in IDsYIT1[first1:last1]],
            'YITDataset3': [z for z in IDsYIT3 if z not in IDsYIT3[first3:last3]] 
        }
        x['testID'] = newTestIDs
        x['trainID'] = newTrainIDs

        torch.save(x,'/media/danny/Projects/yeast-net/CrossValidation/Finetuned Models/New/model_cp%01d.pt' % (i+1))

addIDs()
import pdb
import torch
import shutil
import os




model = 3
dest_path = 'C:/Users/Danny/Desktop/yeast-net/CrossValAcc/Model' + str(model) + '/CellStar/'
orig_path = 'C:/Users/Danny/Desktop/CellStar/Test Images/Cropped/'

cp = torch.load('model_cp3.pt')
testIDs = cp['testID']

if not os.path.isdir(dest_path):
    os.mkdir(dest_path)

for idx in testIDs:
    if idx < 51:
        orig_filename = '/z1/z1_t_000_000_%03d_BF.png' % (idx+1)
        dest_filename = '/z1_t_000_000_%03d_BF.png' % (idx+1)
    elif idx > 101:
        orig_filename = '/z3/z3_t_000_000_%03d_BF.png' % (idx-101)
        dest_filename = '/z3_t_000_000_%03d_BF.png' % (idx-101)
    else:
        orig_filename = '/z2/z2_t_000_000_%03d_BF.png' % (idx-50)
        dest_filename = '/z2_t_000_000_%03d_BF.png' % (idx-50)
    #pdb.set_trace()
    shutil.copy2(orig_path + orig_filename, dest_path + dest_filename)
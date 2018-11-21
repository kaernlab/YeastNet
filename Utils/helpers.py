import numpy as np

def smooth(x,window_len=5,window='flat'):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def accuracy(true_mask, pred_mask):
    IntOfUnion = np.zeros(2)
    true_bg = (true_mask==0)*1
    true_cl = true_mask
    pred_bg = (pred_mask==0)*1
    pred_cl = pred_mask
    ## Calculate IOU
    Union = np.logical_or(true_bg, pred_bg)
    Intersection = np.logical_and(true_bg, pred_bg)
    IntOfUnion[0] = np.sum(Intersection) / np.sum(Union)
    Union = np.logical_or(true_cl, pred_cl)
    Intersection = np.logical_and(true_cl, pred_cl)
    IntOfUnion[1] = np.sum(Intersection) / np.sum(Union)
    PixAccuracy = true_mask[true_mask==pred_mask].size / true_mask.size

    return PixAccuracy, IntOfUnion


    
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

def centreCrop(image, new_size):
    """ Crops center of images into squares

    This method crops grayscale or RGB images. A square of size new_size x new_size
    is cropped out of the middle of the image. Intended as a utility for other methods
    in this class.

    Input:
        image: 1 or 3 channel image to be cropped.
        new_size: desired width and height of square cropped image.

    Outputs:
        cropped_image: cropped image of size new_size x new_size
    """
    h,w = image.shape[-2:]
    if len(image.shape) > 2:
        cropped_image = image[:, :, h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
    else:
        cropped_image = image[h//2 - new_size//2 : h//2 + new_size//2, w//2 - new_size//2 : w//2 + new_size//2 ]
    return cropped_image


    
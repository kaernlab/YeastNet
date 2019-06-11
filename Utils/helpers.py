import numpy as np
import imageio 

def load_image(path):         
    ## Given a path, loads image using imageio's imread method                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    image = imageio.imread(path) 
    return image

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

def centreCrop(image, new_dimensions):
    """ Crops center of images

    This method crops grayscale or RGB images. If only one dimension is passed, 
    a square of size new_size x new_sizeis cropped out of the middle of the image.
    If a tuple is passed, they are treated as the new height and width of the cropped
    image. Intended as a utility for other methods in this project.

    Input:
        image: 1 or 3 channel image to be cropped.
        new_size: desired height and width of cropped image (int or tuple). (height,width)

    Outputs:
        cropped_image: cropped image of size
    """

    if isinstance(new_dimensions, int):
        new_height = new_dimensions
        new_width = new_dimensions
    else:
        new_height = new_dimensions[0]
        new_width = new_dimensions[1]

    h,w = image.shape[-2:]
    if len(image.shape) > 2:
        cropped_image = image[:, :, h//2 - new_height//2 : h//2 + new_height//2, w//2 - new_width//2 : w//2 + new_width//2 ]
    else:
        cropped_image = image[h//2 - new_height//2 : h//2 + new_height//2, w//2 - new_width//2 : w//2 + new_width//2 ]
    return cropped_image


def autoCrop(image):

    h,w = image.shape[-2:]
    if (h % 2 == 1):
        h-=1

    if (w % 2 == 1):
        w-=1

    new_height = 0
    new_width = 0

    for test_height in range(h, 16, -2):
        if (test_height % 16 == 0):
            new_height = test_height
            break

    for test_width in range(w, 16, -2):
        if (test_width % 16 == 0):
            new_width = test_width
            break

    
    cropped_image = centreCrop(image, (new_height, new_width))
    return cropped_image



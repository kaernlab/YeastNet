import numpy
from scipy import misc
from matplotlib import pyplot 

image = misc.imread('..\\NewSegData\\XY2\\z2_t_000_000_001_BF.tif')

print(image.size)

pyplot.imshow(image, cmap=pyplot.cm.gray)  

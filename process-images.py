import numpy
import imageio
import matplotlib.pyplot as plt


#Load all z stacks
image = imageio.imread('../NewSegData/XY2/z1_t_000_000_005_BF.tif')
image1 = imageio.imread('../NewSegData/XY2/z2_t_000_000_005_BF.tif')
image2 = imageio.imread('../NewSegData/XY2/z3_t_000_000_005_BF.tif')

#Rescale images to 0-1
image = numpy.true_divide(image - image.min(), image.max() - image.min())
image1 = numpy.true_divide(image1 - image1.min(), image1.max() - image1.min())
image2 = numpy.true_divide(image2 - image2.min(), image2.max() - image2.min())

#Stack the 3 zstacks into a 3 channels of an rgb image
image3 = numpy.dstack((image,image1,image2))

#Display image
plt.figure()
plt.imshow(image3)  
plt.show()
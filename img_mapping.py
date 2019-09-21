from scipy import ndimage
from PIL import Image
from pylab import *
import numpy as np

im = array(Image.open('tesla-model-s3.jpg').convert('L'))
H = array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
im2 = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))

figure()
gray()
imshow(im2)
show()
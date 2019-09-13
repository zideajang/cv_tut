from PIL import Image
from pylab import *
import numpy as np
import os
pil_im = Image.open('empire.jpeg')
im = np.array(pil_im)

im2 = 255 - im
im3 = (100.0/255) * im - 100
im4 = 255.0 * (im/255.0)**2

figure()
imshow(im)
figure()
imshow(im2)
figure()
imshow(im3)
figure()
imshow(im4)

show()

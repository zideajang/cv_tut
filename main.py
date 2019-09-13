from PIL import Image
from pylab import *
import numpy as np
import os
pil_im = Image.open('empire.jpeg').rotate(45)
im = np.array(pil_im)
imshow(im)
show()

from PIL import Image
from pylab import *
import os
pil_im = Image.open('empire.jpeg')
# print(pil_im)
imshow(array(pil_im))
x = [100,100,400,400]
y = [200,500,200,500]
plot(x,y,'r*')
plot(x[:2],y[:2])
title('draw points and lines')
show()
import sift
from PIL import Image
import os
from numpy import *
from pylab import *
# import cv2


imname = 'tesla-model-s3.jpg'
# img = cv2.imread(imname)
# cv.imshow(img)
im1 = array(Image.open(imname).convert('L'))
# print(im1)
sift.process_image(imname,'tesla-model-s3.sift')
l1,d1 = sift.read_features_from_file('tesla-model-s3.sift')
figure()
gray()
sift.plot_features(im1,l1,circle=True)
show()
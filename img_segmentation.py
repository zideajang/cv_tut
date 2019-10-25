#coding=utf-8
import cv2
import numpy as np

# geometric_shape
# 读取图片

img= cv2.imread('images/geometric_shape.jpg',0)

cv2.imshow('img',img)
cv2.waitKey(0)
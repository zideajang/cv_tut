from PIL import Image
from pylab import *
import numpy as np
import os
import camera

points = loadtxt('house.p3d').T
points = vstack((points,ones(points.shape[1])))

P = hstack((eye(3),array([[0],[0],[-10]])))
cam = camera.Camera(P)
x = cam.project(points)

figure()
plot(x[0],x[1],'k.')
show()

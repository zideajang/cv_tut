from scipy import linalg
from pylab import *
class Camera(object):
    def __init__(self,P):
        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None
    def project(self,X):
        x = dot(self.P,X)
        for i in range(3):
            x[i] /= x[2]
        return x

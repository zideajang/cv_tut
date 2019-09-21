from scipy.ndimage import filters
import numpy as np
def compute_harris_response(im,sigma=3):
    
    imx = zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imy)

    Wxx = filters.gaussian_filter(imx*imx,sigma)
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)

    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet/Wtr

def get_harris_pointers(harrisim,min_dist=10,threshold=0.1):
    
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    coords = array(harrisim_t.nonzero()).T

    candidate_values = [harrisim[c[0]]]
    
    for i in index:


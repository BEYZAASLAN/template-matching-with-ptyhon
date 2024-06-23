import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt 
img = cv.imread('messi5.jpg',0)
img2 = img.copy()
template= cv.imread('messi-face.jpg',0)
w, h = template.shape[:: -1]
methods = ['cv.TM_CCOEFF','cv.TM_CCOEEF_NORMED','cv.TM_CCORR','cv.TM_CCORR_NORMED','cv.TM_SQDIFF','cv.TM_SQDIFF_NORMED']
for meth in methods:
    img=img2.copy()
    method=eval(meth)
    #şABLON EŞLEŞTİRMEYİ UYGULA
    res =cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    #method 

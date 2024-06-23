import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt 
img = cv.imread('messi5.jpg',0)
img2 = img.copy()
template= cv.imread('messi_face.jpg',0)
w, h = template.shape[:: -1]
methods = ['cv.TM_CCOEFF','cv.TM_CCOEEF_NORMED','cv.TM_CCORR','cv.TM_CCORR_NORMED','cv.TM_SQDIFF','cv.TM_SQDIFF_NORMED']
for meth in methods:
    img=img2.copy()
    method=eval(meth)
    #şABLON EŞLEŞTİRMEYİ UYGULA
    res =cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    #method cv.TM_SQDIFF yada  TM_SQDIFF_NORMED ise min al 
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left=min_loc
    else:
        top_left=max_loc
    bottom_right=(top_left[0]+w,top_left[1]+h)
    cv.rectangle(img,top_left, bottom_right,255,2)
    plt.subplot(121),plt.imshow(res,cmap='gray')
    plt.title('Eşleşme Sonucu'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap='gray')
    plt.title('tespit edilen nokta'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()


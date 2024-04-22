import cv2 as cv
import numpy as np

#https://docs.opencv.org/4.x/



#https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html



#https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
img = cv.imread('opencv_files/square.jpg')
img = np.float32(img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)
img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',img)

if cv.waitKey(0):
    cv.destroyAllWindows()
    exit()
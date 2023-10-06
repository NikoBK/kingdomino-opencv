import numpy as np
import cv2 as cv

img = cv.imread("dat/cropped/1.jpg")

assert img is not None, "file could not be read"
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

threshold_value = 50
_, img_bin = cv.threshold(gray, threshold_value, 255, cv.THRESH_BINARY)

cv.imshow("test", img_bin)
cv.waitKey(0)
cv.destroyAllWindows()
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

file = "average_color_map.jpg"
img = cv.imread(file)

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
def green_hsv():
    lower = np.array([40, 115, 115])
    upper = np.array([70, 255, 255])
    mask = cv.inRange(hsv_img, lower, upper)
    return mask

def blue_hsv():
    lower = np.array([95, 100, 100])
    upper = np.array([120, 255, 255])

    mask = cv.inRange(hsv_img, lower, upper)
    return mask


hsv_mask = blue_hsv()
cv.imshow("hsv mask", hsv_mask)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)
cv.imshow("mask", mask)

contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img, contours, -1, (0,0,255), 3)

# for contour in contours:
    #x,y,w,h = cv.drawContours(contour)
    #cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
    #cv.drawContours(img, contour, 0, (0,0,255), 3)

plt.figure(figsize=(10, 8))
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
import cv2
import numpy as np

img = cv2.imread('data/cropped/5.jpg')
hsv_img = cv.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_range = (0, 50, 50)
# Green (HSV): 12, 25, 25
# Red (HSV): 0, 50, 50
upper_range = (10, 255, 255)
mask = cv2.inRange(hsv_image, lower_range, upper_range)

color_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Color Image', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
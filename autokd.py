""" File: autokd.py

Execute this class to run AutoKD!
This project uses PEP 8 naming convention.

Created on: Oct 2, 2023
Authors:
    Gustav Bay Nielsen
    Nikolaj Bjoernager Krebs
"""
from tileparser import TileParser
import numpy as np
import cv2 as cv
import random

class AutoKD:
    def __init__(self):
        self.tileparser = TileParser()
        self.cropped_path = "dat/cropped/"
        self.uncropped_path = "dat/uncropped/"
        self.templates_path = "dat/templates/"

    # Start the automatic score evaluation.
    def start(self):
        # sussy baka images:
        # 6 11 12 13 14 16 
        path = f"{self.cropped_path}27.jpg"
        space = np.zeros([500, 300, 3], np.uint8)
        img = cv.imread(path) # TODO: This should be recursive
        if img is None:
            print(f"Image: {path} could not be found!")
            return

        tiles_color_map = self.tileparser.parse_tiles(img)
        color_map = self.image_resize(tiles_color_map, height=500)
        cv.imwrite(f"./average_color_map_{path}.jpg", color_map)
        contours = self.tileparser.find_contours(color_map)
        print(f"TOTAL CONTOUR COUNT: {len(contours)}")
        for i in range(len(contours)):
            cv.drawContours(img, contours, i, (random.randint(0, 112), 0, random.randint(112, 255)), 3)

        win_title = "Input Image | Average Color Map"
        cv.imshow(win_title, np.hstack([img, space, color_map]))
        cv.waitKey(0)

    def image_resize(self, image, width = None, height = None, inter = cv.INTER_AREA):
        # Initialize the dimensions of the image to be resized and grab the image size.
        dim = None
        (h, w) = image.shape[:2]

        # If both the width and height are None, then return the original img.
        if width is None and height is None:
            return image

        # Check to see if the width is None.
        if width is None:
            # Calculate the ratio of the height and construct the dimensions.
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            # Calculate the ratio of the width and construct the dimensions.
            r = width / float(w)
            dim = (width, int(h * r))

        # Resize the image
        resized = cv.resize(image, dim, interpolation = inter)
        return resized

main = AutoKD()
main.start()
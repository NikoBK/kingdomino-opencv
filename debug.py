import cv2 as cv
import numpy as np
import random

class Debugger:
    def __init__(self):
        self.show_input = False
        self.show_contours = True
        self.show_hsv_values = False
        self.show_gray_scale = False

    def init(self, input_img, dom_col_img, gray_img, hsv_img):
        if self.show_input:
            self.display_input(input_img)
        if self.show_contours:
            self.find_contours(input_img, dom_col_img)
        if self.show_hsv_values:
            self.display_hsv_vals(hsv_img)
        if self.show_gray_scale:
            self.display_grayscale(gray_img)

    def display_input(self, img):
        cv.imshow("[DEBUG] Input Image", img)
        self.wait_for_input()

    def display_hsv_vals(self, img):
        cv.imshow("[DEBUG] Image HSV Values (Hue / Sat / Val)", img)
        self.wait_for_input()

    def display_grayscale(self, img):
        cv.imshow("[DEBUG] Terrain Generation Grayscaled", img)
        self.wait_for_input()

    def wait_for_input(self):
        cv.waitKey(0)
        cv.destroyAllWindows()

    # HSV Thresholding (Hue/Saturation/Value)
    def get_hsv_mask(self, img, terrain):
        match terrain:
            # TODO: Add spawn regions?
            case "forest":
                lower = np.array([30, 50, 0])
                upper = np.array([75, 255, 125])
            case "lake":
                lower = np.array([95, 50, 50])
                upper = np.array([120, 255, 255])
            case "plains":
                lower = np.array([30, 80, 80])
                upper = np.array([95, 255, 255])
            case "wasteland":
                lower = np.array([0, 20, 80])
                upper = np.array([70, 170, 150])
            case "field":
                lower = np.array([25, 120, 120])
                upper = np.array([30, 255, 255])
            case "mine":
                 lower = np.array([0, 0, 0])
                 upper = np.array([28, 180, 80])
            case _:
                lower = np.array([0, 0, 0])
                upper = np.array([0, 0, 0])
                print(f"Could not find terrain handler: {terrain}")
        mask = cv.inRange(img, lower, upper)
        return mask

    def find_contours(self, input_img, dom_col_img):
        terrains = [ "forest", "field" ]
        all_contours = []
        hsv_img = cv.cvtColor(dom_col_img, cv.COLOR_BGR2HSV)
        
        for terrain in terrains:
            hsv_mask = self.get_hsv_mask(hsv_img, terrain)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
            mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                all_contours.append(contour)
        
        for i in range(len(all_contours)):
            cv.drawContours(input_img, all_contours, i, (random.randint(0, 112), 0, random.randint(112, 255)), 3)
        cv.imshow("Input Image (with contours)", input_img)
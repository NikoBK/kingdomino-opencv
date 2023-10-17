import cv2 as cv
import numpy as np
import random

class Debugger:
    def __init__(self):
        # USER-INTERFACE
        self.show_input = False
        self.show_contours = True
        self.show_hsv_values = True
        self.show_gray_scale = False
        # PROPERTIES
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.org = (50,50)
        self.font_scale = 1
        self.color = (255,0,0)
        self.thickness = 2

    def init(self, input_img, dom_col_img, gray_img, hsv_img):
        space = np.zeros([500, 150, 3], np.uint8)
        if self.show_contours and self.show_hsv_values:
            contour_img = input_img
            contour_hsv_img = cv.cvtColor(contour_img, cv.COLOR_BGR2HSV)
            all_contours = self.find_contours(contour_img, dom_col_img)
            for i in range(len(all_contours)):
                cv.drawContours(contour_img, all_contours, i, (random.randint(0, 112), 0, random.randint(112, 255)), 3)

            for y in range(0, dom_col_img.shape[0], 100):
                for x in range(0, dom_col_img.shape[1], 100):
                    dom_color_tile = contour_hsv_img[y:y + 100, x:x + 100,:]
                    print(f"hsv value at y:{int(y * 0.01)}, x:{int(x * 0.01)}: {dom_color_tile[25, 25]}")
                    

            cv.imshow("Contours | HSV Values", np.hstack([contour_img, space, input_img]))
            self.wait_for_input()
            return
        elif self.show_input:
            self.display_input(input_img)
        elif self.show_contours:
            contour_img = input_img
            all_contours = self.find_contours(contour_img, dom_col_img)
            for i in range(len(all_contours)):
                cv.drawContours(contour_img, all_contours, i, (random.randint(0, 112), 0, random.randint(112, 255)), 3)
            cv.imshow("Contours", contour_img)
            self.wait_for_input()
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
        terrains = [ "forest" ]
        all_contours = []
        hsv_img = cv.cvtColor(dom_col_img, cv.COLOR_BGR2HSV)
        
        for terrain in terrains:
            hsv_mask = self.get_hsv_mask(hsv_img, terrain)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
            mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                all_contours.append(contour)
        
        return all_contours
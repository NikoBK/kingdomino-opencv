import cv2 as cv
import numpy as np
import random

class Debugger:
    def __init__(self):
        # LOGGING
        self.verbose = False

        # USER-INTERFACE
        self.show_input = False
        self.show_contours = True
        self.show_hsv_values = True
        self.show_gray_scale = False

        # TEXT/FONT PROPERTIES
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.org = (50,50)
        self.font_scale = 1
        self.color = (255,0,0)
        self.thickness = 2

    # NOTE: **Contours**
    #    Only ever find contours for dom_col_img, and then draw contours on contour_img.
    def init(self, input_img, contour_img, dom_col_img, gray_img, hsv_img):
        # Empty space is used to seperate images in cv.imshow.
        empty_space_height = 500
        empty_space_width = 50
        space = np.zeros([empty_space_height, empty_space_width, 3], np.uint8)

        # 56, 59, 39
        blue = 39
        green = 59
        red = 56
        self.convert_bgr_to_hsv(blue,green,red)

        # Show contours & HSV values
        if self.show_contours and self.show_hsv_values:
            # Get the HSV values for the dominant color matrix.
            dominant_colors_hsv = cv.cvtColor(dom_col_img, cv.COLOR_BGR2HSV)

            # Generate contours on the input image for side-by-side comparison with HSV values.
            all_contours = self.find_contours(dom_col_img)
            for i in range(len(all_contours)):
                cv.drawContours(contour_img, all_contours, i, (random.randint(0, 112), 0, random.randint(112, 255)), 3)

            # Should be 100x100px for each tile on the dominant color matrix img (HSV)
            tile_size = 100
            for y in range(0, dominant_colors_hsv.shape[0], tile_size):
                for x in range(0, dominant_colors_hsv.shape[1], tile_size):
                    dom_color_tile = dominant_colors_hsv[y:y + tile_size, x:x + tile_size,:]
                    if self.verbose:
                        print(f"hsv value at y:{int(y * 0.01)}, x:{int(x * 0.01)}: {dom_color_tile[25, 25]}")
                    

            cv.imshow("Contours | HSV Values", np.hstack([contour_img, space, input_img, space, dom_col_img]))
            self.wait_for_input()
            return
        
        # Show the input image.
        elif self.show_input:
            self.display_input(input_img)

        # Show contours only.
        elif self.show_contours:
            contour_img = input_img
            all_contours = self.find_contours(contour_img)
            for i in range(len(all_contours)):
                cv.drawContours(contour_img, all_contours, i, (random.randint(0, 112), 0, random.randint(112, 255)), 3)
            cv.imshow("Contours", contour_img)
            self.wait_for_input()

        # Show contours only.
        if self.show_hsv_values:
            self.display_hsv_vals(hsv_img)

        # Show grayscale only.
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
                lower = np.array([26, 35, 0])
                upper = np.array([88, 255, 67])
            case "lake":
                lower = np.array([95, 50, 50])
                upper = np.array([120, 255, 255])
            case "plains":
                lower = np.array([30, 80, 78])
                upper = np.array([95, 255, 255])
            case "wasteland":
                lower = np.array([0, 20, 40])
                upper = np.array([70, 150, 150])

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

    def find_contours(self, input_img):
        terrains = []
        check_all = False

        if check_all:
            terrains = [ "forest", "lake", "plains", "wasteland", "field", "mine" ]
        else:
            terrains = [ "forest" ]

        all_contours = []
        hsv_img = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)
        
        for terrain in terrains:
            hsv_mask = self.get_hsv_mask(hsv_img, terrain)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
            mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                all_contours.append(contour)
        
        return all_contours
    
    def convert_bgr_to_hsv(self, b, g, r):
        sample = np.zeros([250, 250, 3], dtype=np.uint8)
        for y in range(0, sample.shape[0], 1):
            for x in range(0, sample.shape[1], 1):
                sample[y,x] = [b,g,r]
        # cv.imshow("test", sample)
        hsv_img = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
        color = hsv_img[25,25]
        h = color[0]
        s = color[1]
        v = color[2]
        print(f"RGB: ({b},{g},{r}) to HSV: ({h},{s},{v})")
""" File: autokd.py

Execute this class to run AutoKD!
This project uses PEP 8 naming convention.

Created on: Oct 2, 2023
Authors:
    Gustav Bay Nielsen
    Nikolaj Bjoernager Krebs
"""
import numpy as np
import cv2 as cv
from debug import Debugger

class Tile:
    def __init__(self):
        self.img = None # 100x100px image of the tile.
        self.dominant_color = [0,0,0]
        self.x = 0
        self.y = 0

class AutoKD:
    def __init__(self):
        self.debugger = Debugger()
        self.tiles = []
        self.input_img = None
        self.contour_img = None
        self.verbose = False
        self.start()
    
    def start(self):
        path = "dat/cropped/40.jpg"
        input_img = cv.imread(path) # Image of the board.
        self.contour_img = cv.imread(path) # DEBUG ONLY
        self.input_img = input_img # Global reference

        # Stop the script if we the path is invalid.
        if input_img is None:
            print(f"[ERROR] No image file found at: {path}")
            return
        else:
            self.dominant_colors(input_img)
        
    def dominant_colors(self, img):
        # A 5x5 matrix holding the most dominant BGR color for each tile.
        dom_col_matrix = np.zeros([5,5,3], dtype=np.uint8)

        # The size of each tile on the board in px.
        tile_size = 100

        for y in range(0, img.shape[0], tile_size):
            for x in range(0, img.shape[1], tile_size):
                tile = Tile()

                # Save the tile's position in the fullscale image.
                tile.x = int(x * 0.01)
                tile.y = int(y * 0.01)

                # Cut out a 100x100px image from the board.
                tile.img = img[y:y + 100, x:x + 100,:]

                # Get the dominant color (BRG)
                pixels = np.float32(tile.img.reshape(-1, 3))

                # Define the ranks of dominant pixels. 5 = top 5 most frequent pixels.
                # This value can be increased if you want to see more and less frequent
                # colors from the image.
                n_colors = 5

                # Look up OpenCV's documentation for kmeans for this part.
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
                flags = cv.KMEANS_RANDOM_CENTERS
                _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)

                # The dominant color taken from the palette.
                dominant_color = palette[np.argmax(counts)]

                if self.verbose:
                    if x == 200 and y == 200:
                        indices = np.argsort(counts)[::-1]   
                        freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
                        rows = np.int_(img.shape[0]*freqs)
                        dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
                        for i in range(len(rows) - 1):
                            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
                        cv.imshow(f"dominant colors ({y},{x})", dom_patch)

                # Save the dominant color in the tile class instance.
                tile.dominant_color = dominant_color

                if self.verbose:
                    print(f"dominate color: {counts}")

                # Append the tile to AutoKD's tiles array, and
                # save the tile's domninant BGR color in the 
                # dominant color matrix.
                self.tiles.append(tile)
                dom_col_matrix[tile.y, tile.x] = [
                    tile.dominant_color[0],
                    tile.dominant_color[1],
                    tile.dominant_color[2]
                ]
        self.make_grayscale(dom_col_matrix)

    # Code by 'thewaywewere' on StackOverflow:
    # https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv/44659589#44659589
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

    # Make a grayscale 5x5 matrix later used for the grassfire algorithm.
    # All properties have their own unique intensity.
    def make_grayscale(self, dom_col_matrix):
        # Construct an empty matrix for the grayscale image.
        gray_img = np.zeros([5,5,1], dtype=np.uint8)

        # All the terrain types we want to find.
        terrains = [ "forest", "lake", "plains", "wasteland", "field", "mine" ]

        # Start intensity is 40 because 0 is black.
        intensity = 40
        # cv.imshow("grayscale img", gray_img)

        # Convert the dominate color matrix to HSV colorspace so we can threshold.
        hsv_img = cv.cvtColor(dom_col_matrix, cv.COLOR_BGR2HSV)
        tile_size = 1

        # Iterate through all terrain types.
        for terrain in terrains:
            # Get the mask (a binary b/w image) for each terrain type.
            hsv_mask = self.get_hsv_mask(hsv_img, terrain)

            # Loop through every 100x100px on the mask image.
            for y in range(0, hsv_mask.shape[0], tile_size):
                for x in range(0, hsv_mask.shape[1], tile_size):
                    if hsv_mask[y,x] > 0:
                        # Insert all found terrain types into the grayscale image.
                        gray_img[int(y), int(x)] = intensity

            # Increase intensity for each terrain type.
            intensity += 40
            
        gray_img_scaled = self.image_resize(gray_img, height=500)
        next_id = 1
        intensity = 40
        for gy in range(gray_img.shape[0]):
            for gx in range(gray_img.shape[1]):
                if gray_img[gy, gx] == intensity:
                    self.grassfire_algorithm(gray_img, (gx, gy), next_id, intensity)
                    next_id += 1
                    intensity += 40
        
        # Scale the color matrix from a 5x5px resolution to a 500x500px resolution (easier to look at)
        color_matrix_scaled = self.image_resize(dom_col_matrix, height=500)
        self.debugger.init(
            self.input_img, 
            self.contour_img, 
            color_matrix_scaled, 
            gray_img_scaled, 
            hsv_img
        )
        cv.waitKey(0)

    def grassfire_algorithm(self, img, coords, index, intensity):
        x,y = coords
        burn_queue = []
        if img[y,x] == intensity:
            burn_queue.append((y,x))
        else:
            print(f"coords does not match intensity.\n Mismatch at: y:{y}, x:{x} with intensity: {intensity}")

        while len(burn_queue) > 0:
            current = burn_queue.pop(0)
            y,x = current

            img[y,x] = index
            if x + 1 < img.shape[1] and img[y, x+1] == intensity:
                burn_queue.append((y, x + 1))
            if y + 1 < img.shape[0] and img[y + 1, x] == intensity:
                burn_queue.append((y + 1, x))
            if x > 0 and img[y, x - 1] == intensity:
                burn_queue.append((y, x - 1))
            if y > 0 and img[y - 1, x] == intensity:
                burn_queue.append((y - 1, x))

        if self.verbose:
            print(f"grassfire img: \n{img}")


main = AutoKD()
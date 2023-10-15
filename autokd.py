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

# NOTE: Feature extraction could be used and then look at more than HSV.

class Tile:
    def __init__(self):
        self.img = None
        self.dominant = [0,0,0]
        self.x = 0
        self.y = 0
        self.crowns = 0

class AutoKD:
    def __init__(self):
        self.tiles = []
        self.start()

    def start(self):
        path = "dat/cropped/1.jpg"
        img = cv.imread(path)
        
        # Make sure the image actually exists.
        if img is None:
            print(f"[ERROR] Could not find image at: {path}")
            return # Stop the script if we cannot load an image.
        else:
            self.find_dominant_colors(img)
        
    def find_dominant_colors(self, img):
        # 5x5 Dominant Color Matrix containing BGR values.
        dom_color_mat = np.zeros([5,5,3], dtype=np.uint8)

        # Each tile on the loaded image is 100x100 pixels.
        tile_size = 100

        # For loops that start at 0 and increase with 100
        # each time out the x and y axis (always y first).
        for y in range(0, img.shape[0], tile_size):
            for x in range(0, img.shape[1], tile_size):
                # Create a new tile for (y,x)
                tile = Tile()

                # Save the tile's position in the fullscale
                # image in the tile class instance.
                tile.x = int(x / 100)
                tile.y = int(y / 100)

                # Save the 100x100 tile image sliced out from the fullscale
                # image in the tile class instance.
                tile.img = img[y:y + tile_size, x:x + tile_size,:]

                # Sum up all the pixels in one variable.
                pixels = np.float32(tile.img.reshape(-1, 3))

                # Define the ranks of dominant pixels. 5 = top 5 most frequent pixels.
                # This value can be increased if you want to see more and less frequent
                # colors from the image.
                n_colors = 5

                max_iterations = 200
                precision = .1
                attempts = 10
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iterations, precision)
                flags = cv.KMEANS_RANDOM_CENTERS
                _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, attempts, flags)
                _, counts = np.unique(labels, return_counts = True)

                # The dominant color.
                dominant = palette[np.argmax(counts)]

                # Save the dominant color in the tile class instance.
                tile.dominant = dominant

                # Fill the 5x5 color matrix with our dominant colors
                # for each 100x100 tile.
                dom_color_mat[tile.y, tile.x] = [
                    dominant[0], # Blue
                    dominant[1], # Green
                    dominant[2]  # Red
                ]
                # Append each tile in the for loop to our tiles array from __init__()
                self.tiles.append(tile)
                self.make_grayscale(dom_color_mat)

    # Get the HSV mask for a terrain with a preset lower and upper HSV thresholds.
    def get_hsv_mask(self, img, terrain):
        match terrain:
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
    
    # Make a grayscale image of the 5x5 dominant color matrix.
    # ( We need this for the grassfire algorithm )
    def make_grayscale(self, dom_matrix):
        # Convert the dominant color matrix to HSV colorspace.
        dom_mat_hsv = cv.cvtColor(dom_matrix, cv.COLOR_BGR2HSV)

        # All connected morphs must have their own intensity.
        # We increase it by 40 each time for convenience.
        start_intensity = 40
        dom_mat_gray = np.zeros([5, 5, 1], dtype=np.uint8)
        terrains = [ "forest", "lake", "plains", "wasteland", "field", "mine"]

        # Run HSV masking for each terrain and add the mask to the 
        # grayscale image so that we can run grassfire on all terrains.
        for terrain in terrains:
            mask = self.get_hsv_mask(dom_mat_hsv, terrain)
            
            # Iterate through all tiles in the HSV mask.
            for y in range(0, mask.shape[0], 1):
                for x in range(0, mask.shape[1], 1):
                    # Color of tile(y,x).
                    color = mask.mean(axis=0).mean(axis=0)
                    # Check if tile(y,x) is white.
                    if color > 0:
                        dom_mat_gray[int(y), int(x)] = start_intensity
            start_intensity += 40

        cv.imshow("final", dom_mat_gray)
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
        print(f"grassfire img: \n{img}")

main = AutoKD()
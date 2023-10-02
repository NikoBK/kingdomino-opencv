import numpy as np
import cv2 as cv

class BoardChecker:
    def __init__(self):
        path = "dat/cropped/1.jpg"
        img = cv.imread(path)

        # Prevent annoying issues with None images
        if img is None:
            print(f"Could not find image: {path}")
            return

        tile_size = 100 # pixels
        avg_col_matrix = np.zeros([5,5,3], dtype = np.uint8)
        space = np.zeros([500, 300, 3], dtype = np.uint8)

        for y in range(0, img.shape[0], tile_size):
            for x in range(0, img.shape[1], tile_size):
                tile = img[y:y + tile_size, x:x + tile_size,:]
                avg_color_row = np.average(tile, axis = 0)
                avg_color = np.average(avg_color_row, axis = 0)
                avg_col_matrix[int(y * 0.01), int(x * 0.01)] = [avg_color[0], avg_color[1], avg_color[2]]

        print(f"Average color matrix is:\n {avg_col_matrix}")
        print(f"Pixel at 3, 4 is: {avg_col_matrix[3, 4]}")

        resized = self.image_resize(avg_col_matrix, height = 500)
        # cv.imshow("image", img)
        # cv.imshow("average color map", resized)
        cv.imshow("Average Tile Color", np.hstack([img, space, resized]))
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

    

checker = BoardChecker()
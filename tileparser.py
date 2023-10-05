""" File: tileparser.py

Created on: Oct 2, 2023
Authors:
    Gustav Bay Nielsen
    Nikolaj Bjoernager Krebs
"""
import cv2 as cv
import numpy as np

class Tile:
    def __init__(self):
        self.img = None # Image of the tile (contains all pixels)
        self.average_color = [0,0,0] # RGB
        self.x = 0
        self.y = 0
        self.crowns = 0
        self.terrain = "None"
        self.is_spawn = False

    def set_pos(self, x, y):
        self.x = int(x * 0.01)
        self.y = int(y * 0.01)

class TileParser:
    def __init__(self):
        self.tile_size = 100 # Pixels
        self.tiles = []
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.font_org = (50, 50)

    def parse_tiles(self, img):
        color_map_matrix = np.zeros([5,5,3], dtype=np.uint8)
        
        for y in range(0, img.shape[0], self.tile_size):
            for x in range(0, img.shape[1], self.tile_size):
                tile = Tile()
                tile.set_pos(x, y)
                tile.img = img[y:y + self.tile_size, x:x + self.tile_size,:]

                avg_color_row = np.average(tile.img, axis=0)
                avg_color = np.average(avg_color_row, axis=0)
                tile.average_color = avg_color

                print(f"tile ({tile.x},{tile.y})'s average color (BGR) is: {tile.average_color}")
                # print(f"tile ({tile.x},{tile.y})'s average color (HSV) is: {average_hsv}")

                # Only used for debugging.
                color_map_matrix[tile.y, tile.x] = [
                    tile.average_color[0], # Red
                    tile.average_color[1], # Green
                    tile.average_color[2]  # Blue
                ]
                self.tiles.append(tile)
        return color_map_matrix

    def get_hsv_thresholds(self, img, terrain):
        match terrain:
            case "forest":
                lower = np.array([30, 0, 0])
                upper = np.array([45, 185, 100])
            case "lake":
                lower = np.array([95, 50, 50])
                upper = np.array([120, 255, 255])
            case "plains":
                lower = np.array([30, 80, 80])
                upper = np.array([95, 255, 255])
            case "spawn_yellow":
                lower = np.array([25, 100, 130])
                upper = np.array([35, 140, 140])
            case "spawn_red":
                lower = np.array([0, 40, 90])
                upper = np.array([25, 80, 120])
            case "spawn_blue":
                lower = np.array([0, 0, 0])
                upper = np.array([100, 120, 200])
            case "spawn_green":
                lower = np.array([32, 80, 95])
                upper = np.array([70, 95, 110])
            case "wasteland":
                lower = np.array([0, 80, 80])
                upper = np.array([70, 150, 120])
            case "wheat_field":
                lower = np.array([25, 120, 120])
                upper = np.array([30, 255, 255])
            case "mine":
                lower = np.array([0, 110, 50])
                upper = np.array([28, 180, 80])
            case _:
                lower = np.array([0, 0, 0])
                upper = np.array([0, 0, 0])
                print(f"Could not find terrain handler: {terrain}")
        
        mask = cv.inRange(img, lower, upper)
        return mask

    def find_contours(self, img):
        terrains_old = [ "forest", "lake", "plains",
                     "spawn_yellow", "spawn_red", "spawn_blue",
                     "spawn_green", "wasteland", "wheat_field", "mine" ]
        terrains = [ "spawn_green" ]
        all_contours = []
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        for terrain in terrains:
            hsv_mask = self.get_hsv_thresholds(hsv_img, terrain)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
            mask = cv.morphologyEx(hsv_mask, cv.MORPH_OPEN, kernel)
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                all_contours.append(contour)
        
        return all_contours
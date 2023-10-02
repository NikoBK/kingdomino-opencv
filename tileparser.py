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
        self.has_crown = False
        self.terrain = "None"

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
                test = self.parse_terrain(tile)
                
                print(f"tile ({tile.x},{tile.y})'s average color is: {tile.average_color}")
                print(f"tile ({tile.x},{tile.y})'s terrain is: {test}")
                # Only used for debugging.
                color_map_matrix[tile.y, tile.x] = [
                    tile.average_color[0], # Red
                    tile.average_color[1], # Green
                    tile.average_color[2]  # Blue
                ]
        return color_map_matrix
    
    def parse_terrain(self, tile):
        blue = int(tile.average_color[0])
        green = int(tile.average_color[1])
        red = int(tile.average_color[2])

        # print(f"red: {red}, green: {green}, blue: {blue}")

        # Check tile colors
        if blue > 18 and green > 139 and red > 96:
        # if red > 28 and green > 148 and blue > 96:
            return "plain"
        elif blue > 120 and green > 73 and red < 56:
            return "lake"
        elif blue < 40 and green < 70 and red < 62:
            return "forest"
        elif blue < 10 and green > 120 and red > 120:
            return "wheat_field"
        elif blue > 20 and green > 111 and red > 95:
            return "village"
        else:
            return "None"
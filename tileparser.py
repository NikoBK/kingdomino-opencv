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
                terrain = self.parse_terrain(tile)
                tile.terrain = terrain
                
                # print(f"tile ({tile.x},{tile.y})'s average color is: {tile.average_color}")
                print(f"tile ({tile.x},{tile.y})'s terrain is: {tile.terrain}")
                if terrain == "INVALID_TILE_TERRAIN_TYPE":
                    print(f"tile ({tile.x},{tile.y})'s average color is: {tile.average_color}")
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

        # Hardcoded spawn colors
        if blue == 78 and green == 136 and red == 137:
            return "yellow_spawn"
        elif blue == 87 and green == 104 and red == 117:
            return "red_spawn"
        elif blue == 71 and green == 104 and red == 102:
            return "green_spawn"
        elif blue == 102 and green == 110 and red == 102:
            return "blue_spawn"

        # Check tile colors
        # Plains
        if blue > 18 and blue < 70 and green > 111 and red > 82 and red < 161:
            return "plain"
        
        # Lake / Ocean / Water
        elif blue > 90 and blue < 173 and green > 63 and red > 3 and red < 56:
            return "lake"
        
        # Forest
        elif blue > 10 and blue < 40 and green > 40 and green < 70 and red > 28 and red < 62:
            return "forest"
        
        # Wheat Fields
        elif blue < 18 and green > 120 and green < 169 and red > 120 and red < 191:
            return "wheat_field"
        
        # Wastelands
        elif blue < 73 and green < 109 and red > 81:
            return "wasteland"
        
        # Mines
        elif blue > 20 and blue < 32 and green > 45 and green < 63 and red > 55 and red < 75:
            return "mine"

        # Default
        else:
            return "INVALID_TILE_TERRAIN_TYPE"
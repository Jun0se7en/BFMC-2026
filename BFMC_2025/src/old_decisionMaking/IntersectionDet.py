import json
import time
import numpy as np
import cv2
from src.decisionMaking.ImageProcessing import ImagePreprocessing
from collections import deque

class IntersectionDetection:
    def __init__(self, config_file):
        self.config = self.load_config_file(config_file)
        self.load_metadata()
        self.prev_intersection = 0
        self.enable_check = False
        self.recent_max_len = deque(maxlen=5)

    def load_metadata(self):
        self.check_thresh = self.config["INTERSECT_DETECTION"]
        self.crop_ratio = float(self.check_thresh["crop_ratio"])
        self.height = self.config["IMAGE_SHAPE"]["height"]
        self.crop_height_value = int(self.height * self.crop_ratio)
        
        self.max_len_thresh = self.check_thresh['max_points_thresh']
        self.gap_thresh = self.check_thresh['gap_thresh']
        self.img_binary = ImagePreprocessing(self.config['LANE_PREPROCESSING'])
        
    @staticmethod
    def load_config_file(config_file):
        with open(config_file, "r") as jsonfile:
            data = json.load(jsonfile)
            # print("Read successful")
        return data

    def find_maximum_connected_line(self, sybinary):
        # Use np.nonzero to get indices of white pixels (value 255)
        rows, cols = np.nonzero(sybinary == 255)
        if rows.size == 0:
            return 0, float("inf"), np.empty((0, 2), dtype=np.int32)
        # Create an array with (x, y) order
        white_pixel_idx = np.column_stack((cols, rows))
        
        # Sort by x-coordinate
        sort_idx = np.argsort(white_pixel_idx[:, 0])
        white_pixel_idx = white_pixel_idx[sort_idx]
        
        # Vectorized grouping by unique x values:
        # Compute counts and the sum of y-values for each unique x
        unique_x, inverse, counts = np.unique(white_pixel_idx[:, 0], return_inverse=True, return_counts=True)
        sum_y = np.bincount(inverse, weights=white_pixel_idx[:, 1])
        mean_y = (sum_y / counts).astype(np.int32)
        
        # Filter groups based on the minimum points threshold
        valid = counts >= self.check_thresh["minimum_points"]
        if not np.any(valid):
            return 0, float("inf"), np.empty((0, 2), dtype=np.int32)
        new_points = np.column_stack((unique_x[valid], mean_y[valid]))
        
        # Split new_points into connected segments based on x-coordinate differences
        x_coords = new_points[:, 0]
        breaks = np.where(np.diff(x_coords) > self.check_thresh["tolerance"])[0] + 1
        segments = np.split(new_points, breaks)
        
        # Select the segment with the maximum number of points
        max_points = max(segments, key=len)
        max_len = len(max_points)
        gap = int(np.max(max_points[:, 1]) - np.min(max_points[:, 1]))
        return max_len, gap, max_points

    def detect(self, sybinary, debug=False):
        max_len, gap, max_points = self.find_maximum_connected_line(sybinary)
        if debug:
            debug_data = {
                "image_size": list(map(int, sybinary.shape)),
                "max_points": [[int(point[0]), int(point[1])] for point in max_points]
            }
            return [max_len, gap], debug_data
        return [max_len, gap], None

    def process_intersection(self, img, show=False):
        # Crop the image from the defined crop height.
        im_cut = img[self.crop_height_value:, :]
        hlane_det = self.img_binary.process_image2(im_cut)
        
        # Detect intersection parameters from the binary image.
        (max_len, gap), debug_info = self.detect(hlane_det)

        if not show:
            return max_len, gap
        else:
            max_points = debug_info["max_points"] if debug_info else []
            for x, y in max_points:
                cv2.circle(img, (x, y + self.crop_height_value), 1, (0, 0, 255), -1)
            return (max_len, gap), img
    
    def intersection_check(self, img, angle, show=False):
        # Process the intersection data.
        if not show:
            max_len, gap = self.process_intersection(img, show=show)
        else:
            data, img = self.process_intersection(img, show=show)
            max_len, gap = data

        self.recent_max_len.append(max_len)

        # Compute the average of the last 3 max_len values
        avg_max_len = int(np.median(list(self.recent_max_len)))
        
        if avg_max_len > 630 and gap > 200 and gap < 300 and abs(angle) < 8:
            self.enable_check = True
        if self.enable_check:
            if self.prev_intersection - avg_max_len > 100:
                self.enable_check = False
                return 'intersection'
            else:
                self.prev_intersection = avg_max_len
                
        return [avg_max_len, gap], self.enable_check

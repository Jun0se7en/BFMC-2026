import cv2
import numpy as np
import time

class LaneKeeping():
    def __init__(self):
        self.HEIGHT_HORIZON = 80
        self.OFFSET_RATIO = 0.65
        self.CHECKPOINT = 115
        self.calc_speed = 0
        self.zero_speed_counter = 0  # Counter for zero speed
        self.speed_counter = 0  # Counter for non zero-speed conditions
        
        self.last_lane_type = None
        self.lane_hold_start_time = None
        self.LANE_HOLD_DURATION = 1.5  # Time (in seconds) to keep previous lane type before switching

    def find_left_right_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        height, width = gray.shape
        black_image = np.copy(image)

        left_boundary = np.argmax(gray > 0, axis=1).astype(float)
        right_boundary = (width - np.argmax(gray[:, ::-1] > 0, axis=1)).astype(float)

        left_boundary[:self.HEIGHT_HORIZON][left_boundary[:self.HEIGHT_HORIZON] == 0] = np.nan
        right_boundary[:self.HEIGHT_HORIZON][right_boundary[:self.HEIGHT_HORIZON] == width] = np.nan

        valid_left = ~np.isnan(left_boundary) & (left_boundary != 0)
        valid_right = ~np.isnan(right_boundary) & (right_boundary != 320)

        left_lane_coords = np.column_stack((left_boundary[valid_left].astype(int), np.where(valid_left)[0]))
        right_lane_coords = np.column_stack((right_boundary[valid_right].astype(int), np.where(valid_right)[0]))

        return left_lane_coords, right_lane_coords, black_image
    
    def find_middle_points_optimized(self, left_points, right_points, image_height, image_width, black_image):
        OFFSET = 180
        OFFSET_RATIO = 0.7
        num_points = 75
        middle_points = []
        STRAIGHT_SPEED = 40
        CURVE_SPEED = 40

        current_lane_type = "both" if len(left_points) >= num_points and len(right_points) >= num_points else (
            "left" if len(left_points) >= num_points else ("right" if len(right_points) >= num_points else "none")
        )

        # Enforce lane hold logic
        if self.last_lane_type in ["left", "right"] and current_lane_type == "both":
            if self.lane_hold_start_time is None:
                self.lane_hold_start_time = time.time()
            elif time.time() - self.lane_hold_start_time < self.LANE_HOLD_DURATION:
                current_lane_type = self.last_lane_type  # Stay in previous lane type for 2 seconds
            else:
                self.lane_hold_start_time = None  # Reset timer after hold period

        self.last_lane_type = current_lane_type  # Update lane type

        if current_lane_type == "both":
            print("Both lanes detected")
            self.calc_speed = STRAIGHT_SPEED
            left_points = np.array(left_points)
            right_points = np.array(right_points)

            left_func = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
            right_func = np.polyfit(right_points[:, 1], right_points[:, 0], 2)

            y_values = np.arange(239, self.HEIGHT_HORIZON, -1)
            x_left = np.polyval(left_func, y_values)
            x_right = np.polyval(right_func, y_values)
            middle_x = (x_left + x_right) / 2

            for i in range(len(y_values)):
                y = y_values[i]
                x1, x2, xm = int(x_left[i]), int(x_right[i]), int(middle_x[i])
                cv2.circle(black_image, (x1, y), 5, (255, 0, 0), -1)
                cv2.circle(black_image, (x2, y), 5, (0, 0, 255), -1)
                cv2.circle(black_image, (xm, y), 5, (0, 255, 0), -1)
                middle_points.append((xm, y))

        elif current_lane_type == "right":
            print("Right lane detected ")
            self.calc_speed = CURVE_SPEED
            right_points = np.array(right_points)
            right_func = np.polyfit(right_points[:, 1], right_points[:, 0], 2)

            y_values = np.arange(239, self.HEIGHT_HORIZON, -1)
            x_right = np.polyval(right_func, y_values)

            for i in range(len(y_values)):
                y = y_values[i]
                x2 = int(x_right[i])
                xm = int(x2 - OFFSET - OFFSET_RATIO * (y - 240 + 1))
                cv2.circle(black_image, (x2, y), 5, (0, 0, 255), -1)
                cv2.circle(black_image, (xm, y), 5, (0, 255, 0), -1)
                middle_points.append((xm, y))

        elif current_lane_type == "left":
            print("Left lane detected")
            self.calc_speed = CURVE_SPEED
            left_points = np.array(left_points)
            left_func = np.polyfit(left_points[:, 1], left_points[:, 0], 2)

            y_values = np.arange(239, self.HEIGHT_HORIZON, -1)
            x_left = np.polyval(left_func, y_values)

            for i in range(len(y_values)):
                y = y_values[i]
                x1 = int(x_left[i])
                xm = int(x1 + OFFSET + OFFSET_RATIO * (y - 240 + 1))
                cv2.circle(black_image, (x1, y), 5, (255, 0, 0), -1)
                cv2.circle(black_image, (xm, y), 5, (0, 255, 0), -1)
                middle_points.append((xm, y))

        return middle_points, black_image

    def AngCal(self, image):
        left, right, image = self.find_left_right_points(image)
        middle_points, black_image = self.find_middle_points_optimized(left, right, 240, 320, image)
        middle_points = np.array(middle_points)
        chosen_middle = self.CHECKPOINT
        angle = 0  

        if len(middle_points) > 0:
            if middle_points[0][1] < chosen_middle:
                x = middle_points[0][0]
                y = middle_points[0][1]
                angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                angle = np.int64(angle/2.3)
                cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
            elif middle_points[-1][1] > chosen_middle:
                x = middle_points[-1][0]
                y = middle_points[-1][1]
                angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                angle = np.int64(angle/2.3)
                cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
            else:
                for i in sorted(middle_points, key=lambda x: x[1]):
                    if i[1] == chosen_middle:
                        x = i[0]
                        y = i[1]
                        angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi)) 
                        angle = np.int64(angle/2.3)
                        cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                        break
        
        angle = max(-25, min(25, -angle))
        cv2.putText(black_image, str(int(angle)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 3)
        
        return self.calc_speed, int(angle), black_image

import cv2
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class VehicleState:
    position: tuple  # (x, y) in pixels
    heading: float   # radians
    speed: float     # pixels/second
    steering_angle: float  # radians

@dataclass
class RoadInfo:
    center_line: np.ndarray
    road_width: float
    curvature: float

class LaneKeeping:
    def __init__(self):
        self.camera_resolution = (320, 240)
        self.max_speed = 20.0
        self.look_ahead_distance = 50  # Reduced from 100
        self.Kp, self.Ki, self.Kd = 0.8, 0.05, 0.1  # Tuned for faster response
        self.integral_error = 0
        self.previous_error = 0
        self.last_time = time.time()
        self.calc_speed = 0

        self.vehicle_state = VehicleState(
            position=(self.camera_resolution[0] // 2, self.camera_resolution[1] - 20),
            heading=np.pi / 2,
            speed=0.0,
            steering_angle=0.0
        )

        # Pre-allocate arrays to avoid repeated allocation
        self.center_line = np.zeros((self.look_ahead_distance, 2), dtype=np.int32)

    def segment_road(self, image):
        """Simplified and faster road segmentation."""
        # Use a smaller ROI (bottom half of the image)
        h, w = image.shape[:2]
        roi = image[h // 2:, :, :]

        # Convert to grayscale instead of HSV (faster)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Light blur for noise reduction

        # Simple threshold for road/lane (assuming bright lanes)
        _, lane_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Fast edge detection
        edges = cv2.Canny(lane_mask, 50, 150, apertureSize=3)

        # Lightweight Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=50)

        # Process lines
        if lines is not None:
            left_x, left_y, right_x, right_y = [], [], [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.1:  # Skip near-horizontal lines
                    continue
                if slope < 0:
                    left_x.extend([x1, x2])
                    left_y.extend([y1 + h // 2, y2 + h // 2])  # Adjust for ROI offset
                else:
                    right_x.extend([x1, x2])
                    right_y.extend([y1 + h // 2, y2 + h // 2])

            # Fit lines if enough points
            if len(left_x) > 2 and len(right_x) > 2:
                left_fit = np.polyfit(left_y, left_x, 1)  # Use linear fit (faster than quadratic)
                right_fit = np.polyfit(right_y, right_x, 1)
                y_points = np.linspace(h - self.look_ahead_distance, h, self.look_ahead_distance, dtype=np.int32)
                for i, y in enumerate(y_points):
                    left_x = int(left_fit[0] * y + left_fit[1])
                    right_x = int(right_fit[0] * y + right_fit[1])
                    center_x = (left_x + right_x) // 2
                    self.center_line[i] = [center_x, y]
                road_width = np.abs(right_x - left_x)
                curvature = 10000  # Simplified: assume straight unless tuned later
            else:
                # Fallback: straight path
                road_width = 80
                curvature = 10000
                y_points = np.linspace(h - self.look_ahead_distance, h, self.look_ahead_distance, dtype=np.int32)
                for i, y in enumerate(y_points):
                    self.center_line[i] = [w // 2, y]
        else:
            road_width = 80
            curvature = 10000
            y_points = np.linspace(h - self.look_ahead_distance, h, self.look_ahead_distance, dtype=np.int32)
            for i, y in enumerate(y_points):
                self.center_line[i] = [w // 2, y]

        return RoadInfo(self.center_line, road_width, curvature)

    def calculate_target_path(self, road_info):
        """Fast target point selection."""
        look_ahead_idx = self.look_ahead_distance // 2
        return road_info.center_line[look_ahead_idx]

    def calculate_steering_angle(self, target_point):
        """Optimized PID steering calculation."""
        dx = target_point[0] - self.vehicle_state.position[0]
        dy = self.vehicle_state.position[1] - target_point[1]
        target_angle = np.arctan2(dy, dx)
        error = target_angle - self.vehicle_state.heading
        error = np.arctan2(np.sin(error), np.cos(error))

        current_time = time.time()
        dt = max(current_time - self.last_time, 0.001)
        self.last_time = current_time
        self.integral_error = min(max(self.integral_error + error * dt, -1.0), 1.0)  # Clamp integral
        derivative = (error - self.previous_error) / dt
        self.previous_error = error

        steering_angle = self.Kp * error + self.Ki * self.integral_error + self.Kd * derivative
        return np.clip(steering_angle, -np.pi / 4, np.pi / 4)

    def calculate_speed(self, road_info):
        """Simplified speed calculation."""
        speed = self.max_speed * (1.0 - abs(self.vehicle_state.steering_angle) / (np.pi / 4))
        self.calc_speed = max(self.max_speed * 0.5, min(speed, self.max_speed))
        return self.calc_speed

    def update_vehicle_state(self, dt, steering_angle, speed):
        """Fast vehicle state update."""
        self.vehicle_state.steering_angle += (steering_angle - self.vehicle_state.steering_angle) * 0.7  # Faster response
        self.vehicle_state.speed += (speed - self.vehicle_state.speed) * 0.5
        wheelbase = 30
        self.vehicle_state.heading += (self.vehicle_state.speed * np.tan(self.vehicle_state.steering_angle) / wheelbase) * dt
        self.vehicle_state.heading = np.arctan2(np.sin(self.vehicle_state.heading), np.cos(self.vehicle_state.heading))
        self.vehicle_state.position = (
            self.vehicle_state.position[0] + self.vehicle_state.speed * np.sin(self.vehicle_state.heading) * dt,
            self.vehicle_state.position[1] - self.vehicle_state.speed * np.cos(self.vehicle_state.heading) * dt
        )

    def AngCal(self, image):
        """Optimized main function."""
        try:
            road_info = self.segment_road(image)
            black_image = image.copy()

            target_point = self.calculate_target_path(road_info)
            steering_angle_rad = self.calculate_steering_angle(target_point)
            steering_angle_deg = float(np.degrees(steering_angle_rad))
            speed = self.calculate_speed(road_info)

            dt = 0.033  # ~30 FPS
            self.update_vehicle_state(dt, steering_angle_rad, speed)

            # Minimal visualization
            cv2.polylines(black_image, [road_info.center_line], False, (0, 255, 0), 1)
            cv2.circle(black_image, (int(self.vehicle_state.position[0]), int(self.vehicle_state.position[1])), 3, (255, 0, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(black_image, f"{steering_angle_deg:.1f}", (10, 20), font, 0.5, (0, 0, 255), 2)
            cv2.putText(black_image, f"Speed: {speed:.1f}", (10, 40), font, 0.5, (0, 0, 255), 2)
            print("Angle: ", steering_angle_deg, "Speed: ", speed)
            return speed, steering_angle_deg, black_image
        except Exception as e:
            print(f"Error: {e}")
            return 0, 0.0, image

if __name__ == "__main__":
    lane_keeper = LaneKeeping()
    cap = cv2.VideoCapture('capture.mp4')  # Replace with your video source
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        speed, angle, viz_image = lane_keeper.AngCal(frame)
        process_time = time.time() - start_time
        print(f"Frame time: {process_time:.3f}s, FPS: {1/process_time:.1f}")

        cv2.imshow('Lane Keeping', viz_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
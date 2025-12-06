import cv2
import threading
import base64
import time
import numpy as np
import os
import sys
import json
import random
import ctypes
import pandas as pd
from scipy.optimize import minimize

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Record,
    Config,
    Position,
    Heading,
    SendHeading
)
from src.templates.threadwithstop import ThreadWithStop
import socket
import json
import math
import signal
################ EKF ############################
from lib.IMU_GPS_EKF_5States import ExtendedKalmanFilter
# from lib.EKF_test1 import ExtendedKalmanFilter
from lib.Khoi_EKF import PositionEKF
from lib.Test_EKF import IMUEKF
from lib.EKF_IMU_GPS import EKFImuGps
##################################################
from lib.Transform import Transform
import serial
import pynmea2
import struct
from scipy.spatial.transform import Rotation as R

## MPC ##
from lib.MPC_kinematic_bycicle_model import KinematicBicycleModel
from src.utils.CarControl.CarControl import CarControl

def integrate_xy(v, heading_deg, x, y, vx, vy, ax, ay, dt=0.1):
    """Trả về (x, y) cho toàn bộ chuỗi"""
    heading_rad = math.radians(heading_deg)
    vx = vx + ax * dt
    vy = vy + ay * dt
    x = x + vx * np.cos(heading_rad) * dt
    y = y + vy * np.sin(heading_rad) * dt
    return x, y, vx, vy

def wrap180(angle_deg):
    # trả về (-180, 180]
    while angle_deg <= -180: angle_deg += 360
    while angle_deg >  180: angle_deg -= 360
    return angle_deg

def heading_math_to_wtgahrs(theta_math_deg):
    """
    Đổi góc CCW từ +X (chuẩn toán/Matplotlib) → chuẩn WTGAHRS2:
    0° = +Y, CW+, phạm vi (-180, 180].
    """
    return wrap180(90 - theta_math_deg)

def compute_offset(A, B, imu_raw_deg):
    """
    Tính offset giữa IMU và thực địa.
    - A, B: tuple/list (x, y) trên sơ đồ Matplotlib
    - imu_raw_deg: giá trị IMU báo ngay tại vị trí A (thường = 0 sau khi reboot)
    Trả về offset_deg cần cộng vào mọi giá trị IMU sau này.
    """
    dx, dy = B[0] - A[0], B[1] - A[1]
    theta_math = np.degrees(np.arctan2(dy, dx))       # CCW từ +X
    theta_ref  = heading_math_to_wtgahrs(theta_math)  # chuyển sang chuẩn WTGAHRS2
    return wrap180(theta_ref - imu_raw_deg)           # sai lệch cần bù

def apply_offset(imu_raw_deg, offset_deg):
    """Trả về heading đã hiệu chuẩn, vẫn ở chuẩn WTGAHRS2 (-180…180)."""
    return wrap180(imu_raw_deg + offset_deg)

def wtgahrs_to_math(heading_deg):
    """Chuyển ngược: WTGAHRS2 → CCW từ +X (để vẽ mũi tên Matplotlib)."""
    return wrap180(90 - heading_deg)

################### Calculate Steering ############################

def find_closest_and_next_points(current_position, reference_path, horizon = 3):
    """
    Find the closest reference point and the next target for MPC.
    
    Args:
        current_position (tuple): The current (x, y) position.
        reference_path (list): A list of reference points [(x1, y1), (x2, y2), ...].
        
    Returns:
        tuple: The closest point and the next target point.
    """
    distances = [np.linalg.norm(np.array(current_position) - np.array(point)) for point in reference_path]
    closest_index = np.argmin(distances)
    closest_point = reference_path[closest_index]
    if closest_index + 1 > len(reference_path):
        return None
    if closest_index + horizon > len(reference_path):
        return closest_point, reference_path[closest_index + 1:]
    else:
        return closest_point, reference_path[closest_index + 1: closest_index+horizon+1]

def find_angle(current_position, target_positions):
    """
    Calculate the angle (in radians) at current_position formed by the two vectors:
    current_position -> target_positions[0] and current_position -> target_positions[1]
    
    Args:
        current_position (tuple): The current (x, y) position.
        target_positions (tuple): Two target positions ((x1, y1), (x2, y2)).
        
    Returns:
        float: The angle in radians.
    """
    assert len(target_positions) == 2, "target_positions must contain exactly two points."
    
    ax, ay = current_position
    bx, by = target_positions[0]
    cx, cy = target_positions[1]
    
    # Vector BA and BC
    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])
    
    # Compute the cosine of the angle
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # Avoid division by zero
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    # Clamp the cosine value to [-1, 1] to avoid NaN from arccos due to floating point error
    cos_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    return math.degrees(angle_rad)
    
def head_to_target(current_position, target_position):
    """
    Calculate the heading angle in degrees from current_position to target_position.
    
    Args:
        current_position (tuple): The current (x, y) position.
        target_position (tuple): The target (x, y) position.
        
    Returns:
        float: The heading angle in degrees.
    """
    dx = target_position[0] - current_position[0]
    dy = target_position[1] - current_position[1]
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)

def find_steering_angle(current_position, closest_point, next_target_point, current_heading):
    """
    Calculate the steering angle based on the current position, closest point, next target point, and current heading.
    
    Args:
        current_position (tuple): The current (x, y) position.
        closest_point (tuple): The closest reference point (x, y).
        next_target_point (tuple): The next target point (x, y).
        current_heading (float): The current heading in degrees.
        
    Returns:
        float: The calculated steering angle in degrees.
    """
    ac = np.array([next_target_point[0] - current_position[0], next_target_point[1] - current_position[1]])
    # xac = np.arccos(ac[0] / np.linalg.norm(ac)) if np.linalg.norm(ac) != 0 else 0
    # xac = wrap180(xac)
    xac = np.arctan2(ac[1], ac[0])
    beta_xac = xac - np.radians(current_heading)
    
    ab = np.array([closest_point[0] - current_position[0], closest_point[1] - current_position[1]])
    # xab = np.arccos(ab[0] / np.linalg.norm(ab)) if np.linalg.norm(ab) != 0 else 0
    # xab = wrap180(xab)
    xab = np.arctan2(ab[1], ab[0])
    alpha_xab = xab - np.radians(current_heading)
    
    bc = np.array([next_target_point[0] - closest_point[0], next_target_point[1] - closest_point[1]])

    norm_ab = np.linalg.norm(ab)

    if norm_ab <= 20:
        coef = 1
    elif norm_ab >= 40:
        coef = 0
    else:
        coef = 1-((norm_ab-20)/20)
    
    steering = alpha_xab + (beta_xac - alpha_xab) * coef
    
    return math.degrees(steering), math.degrees(xab), math.degrees(xac)


class threadWitMotion(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, Speed, Steer, logger, debugger):
        super(threadWitMotion, self).__init__()
        self.queuesList = queuesList

        self.Speed = Speed
        self.Steer = Steer

        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.debugger = debugger
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        # self.client.send(self.speed, self.smooth_angle)
        self.subscribe()
        self.Configs()
        self.message = {}
        self.message_type = ""
        # self.csv_data = {
        #     "X": [],
        #     "Y": [],
        #     "Acc_x": [],
        #     "Acc_y": [],
        #     "heading": [],
        #     "transformed_heading": [],
        # }
    
        ### Raw ###
        self.csv_data = {
            "Acc_x": [],
            "Acc_y": [],
            "Acc_z": [],
            "Gyro_x": [],
            "Gyro_y": [],
            "Gyro_z": [],
            "Angle_x": [],
            "Angle_y": [],
            "Angle_z": [],
            "Time": [],
        }
        ############# EKF TEST ###############
        # self.ekf = PositionEKF()
        # self.ekf = IMUEKF()
        # self.ekf = PosHeadingEKF()
        self.ekf = EKFImuGps()
        ######################################
        self.transform = Transform()
        self.initialize = False

        self.SERIAL_PORT = '/dev/ttyUSB0'        # Windows
        self.BAUDRATE = 115200  # thường GPS module là 9600
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=1)

        self.prev_time = 0.0
        self.prev_lat = 0.0
        self.prev_lon = 0.0
        self.gps_thres = 0.0
        # State biến cho acc
        self.bias = [0.0, 0.0, 0.0]
        self.filt = [0.0, 0.0, 0.0]
        self.calib_idx = 0
        self.CALIB_SAMPLES = 200
        self.ALPHA = 0.135
        self.DEADBAND_G = 0.02

        self.current = {}
        self.DISTANCE_THRESHOLD = 8  # Khoảng cách ngưỡng để cập nhật vị trí mới

        ################## HEADING OFFSET ######################
        self.A = (24, 34)
        self.B = (34, 34)
        self.init_heading_offset = False

        self.GYRO_THRES = 5

        ################### Kiet Algorithm #####################
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        self.df = pd.read_csv('./route_points.csv')
        self.ref_path = list(zip(self.df['x'].values, self.df['y'].values))
        ################### MPC ################################
        # self.route = load_route('./route_points.csv')
        # self.mpc = MPC(route_points=self.route, horizon=3, dt=0.1)
        # self.v = 5
        # self.delta_prev = 0.0
        # self.x = 0
        # self.y = 0

    # ================================= SUBCRIBE ===============================================
    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadWitMotion", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadWitMotion", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        self.ser.close()
        super(threadWitMotion, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    # =============================== UTILITIES ============================================
    def parse_data(self, data):
        if len(data) >= 11 and data[0] == 0x55:
            data_type = data[1]

            if data_type == 0x51:  # Acceleration
                ax, ay, az = struct.unpack('<hhh', data[2:8])
                ax, ay, az = [x / 32768.0 * 16 for x in (ax, ay, az)]
                return 'ACC', (ax, ay, az)

            elif data_type == 0x52:  # Gyroscope
                gx, gy, gz = struct.unpack('<hhh', data[2:8])
                gx, gy, gz = [x / 32768.0 * 2000 for x in (gx, gy, gz)]
                return 'GYRO', (gx, gy, gz)

            elif data_type == 0x53:  # Euler angles (Roll, Pitch, Yaw)
                roll, pitch, yaw = struct.unpack('<hhh', data[2:8])
                roll, pitch, yaw = [x / 32768.0 * 180 for x in (roll, pitch, yaw)]
                return 'ANGLE', (roll, pitch, yaw)

            elif data_type == 0x57:  # GPS Longitude and Latitude
                lon, lat = struct.unpack('<ii', data[2:10])
                lon, lat = lon / 1e7, lat / 1e7
                return 'GPS', (lat, lon)

            elif data_type == 0x58:  # GPS Height and Speed
                height, speed = struct.unpack('<hh', data[2:6])
                height = height / 10.0
                speed = speed / 1000.0 * 3.6
                return 'GPS_SPEED', (height, speed)

            elif data_type == 0x50:  # Time data
                year, month, day, hour, minute, second, ms = struct.unpack('<BBBBBBH', data[2:10])
                year += 2000
                return 'TIME', (year, month, day, hour, minute, second, ms)

        return None, None
    
    def save_data_to_csv(self, csv_file):
        df = pd.DataFrame(self.csv_data)

        df.to_csv(csv_file, mode='a', index=False, header=False)

        for key in self.csv_data.keys():
            self.csv_data[key].clear()

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies and then it send the data to process gateway."""   
        self.data = {
            "ACC": (0, 0, 0),
            "GYRO": (0, 0, 0),
            "ANGLE": (0, 0, 0),
            "TIME": 0.0,
        }
        self.count = 0
        self.full = False
        buffer = bytearray()

        # self.reset_stm()
        # self.reset_stm()

        while self._running:
            byte = self.ser.read()
            # print("Reading")
            if byte:
                buffer += byte

                if buffer[0] != 0x55:
                    buffer.pop(0)
                    continue

                if len(buffer) >= 11:
                    sensor_type, values = self.parse_data(buffer[:11])
                    if sensor_type:
                        if sensor_type == 'ACC':
                            # if self.calib_idx < self.CALIB_SAMPLES:
                            #     for i in range(3):
                            #         self.bias[i] += values[i]
                            #     self.calib_idx += 1
                            #     if self.calib_idx == self.CALIB_SAMPLES:
                            #         self.bias[:] = [b / self.CALIB_SAMPLES for b in self.bias]
                            #         print(f'Bias Calibrated: {self.bias}')
                            #     continue
                            
                            # acc_corr = [values[i] - self.bias[i] for i in range(3)]

                            # # ---- (2) IIR low-pass ----
                            # for i in range(3):
                            #     self.filt[i] = self.ALPHA*acc_corr[i] + (1-self.ALPHA)*self.filt[i]

                            # # ---- (3) Dead-band ----
                            # acc_out = [0.0 if abs(f) < self.DEADBAND_G else f for f in self.filt]
                            
                            # ###### REMOVED ########
                            # # ---- (4) Scale ----
                            # acc_out = [f * 9.81 for f in acc_out]
                            # # # ---- (5) Convert to pixel/s^2 ----
                            # # acc_out = [f * 135/2 for f in acc_out]
                            # acc_out = [f * 9.81 for f in values]
                            self.data[sensor_type] = values
                            self.full = False
                        elif sensor_type == 'GYRO':
                            # print(f"Gyroscope [°/s]: {values}")
                            self.data[sensor_type] = values
                            self.full = False
                        elif sensor_type == 'ANGLE':
                            # print(f"Euler angles [°]: {values}")
                            if not self.init_heading_offset:
                                self.heading_offset = 0 - values[2]
                                # self.heading_offset = compute_offset(self.A, self.B, values[2])
                                self.init_heading_offset = True
                            self.data[sensor_type] = values
                            self.full = False
                        elif sensor_type == 'TIME':
                            year, month, day, hour, minute, second, ms = values
                            self.data[sensor_type] = float(hour) * 3600 + float(minute) * 60 + float(second) + float(ms)/1000
                            self.full = True
                        # else:
                        #     self.full = True
                    buffer = buffer[11:]
            if self.full:
                ######## EKF ########
                # if not self.initialize:
                #     self.initialize = True
                #     self.prev_lat = 0
                #     self.prev_lon = 0
                #     self.x = 0
                #     self.y = 0
                #     self.vx = 0
                #     self.vy = 0
                #     # self.ekf.state = [0, 0, 0.0, 0.0]
                #     # self.kbm = BicycleModel(wheelbase=0.195, x0=self.x, y0=self.y, yaw0=0, v0=0)
                #     self.prev_time = self.data["TIME"]
                # else:
                #     dt = (self.data["TIME"] - self.prev_time)
                #     self.prev_time = self.data["TIME"]
                #     ### EKF ###
                #     if (self.data["ACC"][0] != 0 or self.data["ACC"][1] != 0) or (self.data["GYRO"][0]**2 + self.data["GYRO"][1]**2 + self.data["GYRO"][2]**2 > self.GYRO_THRES):
                #         # self.ekf.predict(self.data["ACC"][0], self.data["ACC"][1], self.data["ACC"][2], self.data["ANGLE"][0], self.data["ANGLE"][1], self.data["ANGLE"][2], dt)
                #         # self.x, self.y, _, _ = self.kbm.step(self.data["ACC"][0], self.data["ACC"][1], self.data["ACC"][2], self.data["GYRO"][0], self.data["GYRO"][1], self.data["GYRO"][2], self.data["ANGLE"][0], self.data["ANGLE"][1], self.data["ANGLE"][2], dt)
                #         self.x, self.y, self.vx, self.vy = integrate_xy(10, self.data["ANGLE"][2], self.x, self.y, self.vx, self.vy, self.data["ACC"][0], self.data["ACC"][1], dt)
                #     # elif self.data["ACC"][0] == 0.0 and self.data["ACC"][1] == 0.0:
                #     #     self.vx = 0
                #     #     self.vy = 0
                #     ### SEND DATA ###
                #     # self.x = self.ekf.state[0]
                #     # self.y = self.ekf.state[1]
                #     print("EKF: ", self.x, self.y)
                #     ### Save Data to CSV ###
                #     self.csv_data['X'].append(self.x)
                #     self.csv_data['Y'].append(self.y)
                #     self.csv_data['Acc_x'].append(self.data["ACC"][0])
                #     self.csv_data['Acc_y'].append(self.data["ACC"][1])
                #     self.csv_data['heading'].append(self.data["ANGLE"][2])
                #     self.csv_data['transformed_heading'].append(self.data["ANGLE"][2])
                #     output_csv_path = './EKF_Data.csv'
                #     self.save_data_to_csv(output_csv_path)
                
                dt = (self.data["TIME"] - self.prev_time)
                self.prev_time = self.data["TIME"]
                if dt != 0:
                    if not self.queuesList[Heading.Queue.value].empty():
                        _ = self.queuesList[Heading.Queue.value].get()
                    self.queuesList[Heading.Queue.value].put(
                        {
                            "Owner": Heading.Owner.value,
                            "msgID": Heading.msgID.value,
                            "msgType": Heading.msgType.value,
                            "msgValue": self.data["ANGLE"][2],
                        }
                    )
                    if not self.queuesList[SendHeading.Queue.value].empty():
                        _ = self.queuesList[SendHeading.Queue.value].get()
                    self.queuesList[SendHeading.Queue.value].put(
                        {
                            "Owner": SendHeading.Owner.value,
                            "msgID": SendHeading.msgID.value,
                            "msgType": SendHeading.msgType.value,
                            "msgValue": self.data["ANGLE"][2],
                        }
                    )

                ##### GPS Only ##########
                # gps = {"x": -1.0, "y": -1.0, "z": 0.0, "quality": 0}
                # if not self.queuesList["CarGPSInfo"].empty():
                #     gps = self.queuesList["CarGPSInfo"].get()['msgValue']

                # if (gps["x"] != -1.0 and gps["y"] != -1.0):
                #     data = {
                #         "x": gps["x"],
                #         "y": gps["y"],
                #         "z": gps["z"],
                #         "quality": gps["quality"],
                #         "heading": self.data["ANGLE"][2],
                #         "type": "location",
                #         "id": 3,
                #     }
                #     if not self.queuesList[Position.Queue.value].empty():
                #         _ = self.queuesList[Position.Queue.value].get()
                #     self.queuesList[Position.Queue.value].put(
                #     {
                #         "Owner": Position.Owner.value,
                #         "msgID": Position.msgID.value,
                #         "msgType": Position.msgType.value,
                #         "msgValue": data,
                #     })
                #     try:
                #         closest_point, next_target_points = find_closest_and_next_points((gps["x"], gps["y"]), self.ref_path, horizon=3)
                #         angle = find_angle((gps["x"], gps["y"]), [closest_point, next_target_points[0]])
                #         if angle < 120:
                #             self.ref_path.remove(closest_point)
                #             closest_point, next_target_points = find_closest_and_next_points((gps["x"], gps["y"]), self.ref_path, horizon=3)
                #         ab = np.array([closest_point[0] - gps["x"], closest_point[1] - gps["y"]])
                #         if np.linalg.norm(ab) < 15:
                #             self.ref_path.remove(closest_point)
                #             closest_point, next_target_points = find_closest_and_next_points((gps["x"], gps["y"]), self.ref_path, horizon=3)
                        
                #         steering, tmp1, tmp2 = find_steering_angle((gps["x"], gps["y"]), closest_point, next_target_points[0], wrap180(self.data["ANGLE"][2] + self.heading_offset))
                #         steering = int(np.clip(steering, -25, 25))
                #         print("Offset:", self.heading_offset)
                #         print("Current Position:", gps["x"], gps["y"])
                #         print("Next Target 1:", closest_point)
                #         print("Next Target 2:", next_target_points[0])
                #         print("Steering: ",-steering)
                #         print("XAB:", tmp1, "XAC", tmp2)
                #         print("Heading:", self.data["ANGLE"][2])
                #         print("Transformed Heading:", wrap180(self.data["ANGLE"][2] + self.heading_offset))
                #         self.control.setAngle(-steering)
                #     except:
                #         # print("Out Range!!!!")
                #         continue
                # # print(gps)
            ################# EKF #####################
            # byte = self.ser.read()
            # # print("Reading")
            # if byte:
            #     buffer += byte

            #     if buffer[0] != 0x55:
            #         buffer.pop(0)
            #         continue

            #     if len(buffer) >= 11:
            #         sensor_type, values = self.parse_data(buffer[:11])
            #         if sensor_type:
            #             if sensor_type == 'ACC':
            #                 if self.calib_idx < self.CALIB_SAMPLES:
            #                     for i in range(3):
            #                         self.bias[i] += values[i]
            #                     self.calib_idx += 1
            #                     if self.calib_idx == self.CALIB_SAMPLES:
            #                         self.bias[:] = [b / self.CALIB_SAMPLES for b in self.bias]
            #                         print(f'Bias Calibrated: {self.bias}')
            #                     continue
                            
            #                 acc_corr = [values[i] - self.bias[i] for i in range(3)]

            #                 # ---- (2) IIR low-pass ----
            #                 for i in range(3):
            #                     self.filt[i] = self.ALPHA*acc_corr[i] + (1-self.ALPHA)*self.filt[i]

            #                 # ---- (3) Dead-band ----
            #                 acc_out = [0.0 if abs(f) < self.DEADBAND_G else f for f in self.filt]
                            
            #                 ###### REMOVED ########
            #                 # ---- (4) Scale ----
            #                 acc_out = [f * 9.81 for f in acc_out]
            #                 # ---- (5) Convert to pixel/s^2 ----
            #                 acc_out = [f * 135/2 for f in acc_out]
            #                 self.data[sensor_type] = acc_out
            #                 self.full = False
            #             elif sensor_type == 'GYRO':
            #                 # print(f"Gyroscope [°/s]: {values}")
            #                 self.data[sensor_type] = values
            #                 self.full = False
            #             elif sensor_type == 'ANGLE':
            #                 # print(f"Euler angles [°]: {values}")
            #                 if not self.init_heading_offset:
            #                     self.heading_offset = 0 - values[2]
            #                     # self.heading_offset = compute_offset(self.A, self.B, values[2])
            #                     self.init_heading_offset = True
            #                 self.data[sensor_type] = values
            #                 self.full = False
            #             elif sensor_type == 'TIME':
            #                 year, month, day, hour, minute, second, ms = values
            #                 self.data[sensor_type] = float(hour) * 3600 + float(minute) * 60 + float(second) + float(ms)/1000
            #                 self.full = True
            #             # else:
            #             #     self.full = True
            #         buffer = buffer[11:]
            # # print(self.data)
            # if self.full:
            #     gps = {"x": -1.0, "y": -1.0, "z": 0.0, "quality": 0}
            #     if not self.queuesList["CarGPSInfo"].empty():
            #         gps = self.queuesList["CarGPSInfo"].get()['msgValue']

            #         if gps["quality"] < self.gps_thres:
            #             gps = {"x": -1.0, "y": -1.0, "z": 0.0, "quality": 0}

            #     ###############################################################################################

            #     if not self.initialize:
            #         if gps["x"] != -1.0 and gps["y"] != -1.0:
            #             self.initialize = True
            #             self.prev_lat = gps["x"]
            #             self.prev_lon = gps["y"]
            #             self.ekf.state = [gps["x"], gps["y"], 0.0, 0.0]
            #             self.prev_time = self.data["TIME"]

            #             data = {
            #                 "x": self.ekf.state[0],
            #                 "y": self.ekf.state[1],
            #                 "z": gps["z"],
            #                 "quality": gps["quality"],
            #                 "heading": self.data["ANGLE"][2],
            #                 "transformed_heading": wrap180(self.data["ANGLE"][2] + self.heading_offset),
            #                 "type": "location",
            #                 "id": 3,
            #             }
            #             if not self.queuesList[Position.Queue.value].empty():
            #                 _ = self.queuesList[Position.Queue.value].get()
            #             self.queuesList[Position.Queue.value].put(
            #             {
            #                 "Owner": Position.Owner.value,
            #                 "msgID": Position.msgID.value,
            #                 "msgType": Position.msgType.value,
            #                 "msgValue": data,
            #             })
            #             self.ekf.update_position(gps["x"], gps["y"])
            #     else:
            #         dt = (self.data["TIME"] - self.prev_time) * 1.5
            #         self.prev_time = self.data["TIME"]
            #         ### EKF ###
            #         if (self.prev_lat != gps["x"] and self.prev_lon != gps["y"]) and (gps["x"] != -1.0 and gps["y"] != -1.0) and (math.sqrt((gps["x"] - self.prev_lat)**2 + (gps["y"] - self.prev_lon)**2) > self.DISTANCE_THRESHOLD):
            #             self.ekf.predict(self.data["ACC"][0], self.data["ACC"][1], self.data["ACC"][2], self.data["ANGLE"][0], self.data["ANGLE"][1], wrap180(self.data["ANGLE"][2] + self.heading_offset), dt)
            #             ### TEST ###
            #             # print("Update GPS")
            #             self.ekf.update_position(gps["x"], gps["y"])
            #             ############
            #             self.prev_lat = gps["x"]
            #             self.prev_lon = gps["y"]
            #         elif (self.data["ACC"][0] != 0 or self.data["ACC"][1] != 0) or (self.data["GYRO"][0]**2 + self.data["GYRO"][1]**2 + self.data["GYRO"][2]**2 > self.GYRO_THRES):
            #             self.ekf.predict(self.data["ACC"][0], self.data["ACC"][1], self.data["ACC"][2], self.data["ANGLE"][0], self.data["ANGLE"][1], wrap180(self.data["ANGLE"][2] + self.heading_offset), dt)
            #         ### SEND DATA ###
            #         data = {
            #             "x": self.ekf.state[0],
            #             "y": self.ekf.state[1],
            #             "z": gps["z"],
            #             "quality": gps["quality"],
            #             "heading": self.data["ANGLE"][2],
            #             "transformed_heading": wrap180(self.data["ANGLE"][2] + self.heading_offset),
            #             "type": "location",
            #             "id": 3,
            #         }
            #         if not self.queuesList[Position.Queue.value].empty():
            #             _ = self.queuesList[Position.Queue.value].get()
            #         self.queuesList[Position.Queue.value].put(
            #         {
            #             "Owner": Position.Owner.value,
            #             "msgID": Position.msgID.value,
            #             "msgType": Position.msgType.value,
            #             "msgValue": data,
            #         })
            #         # print("EKF: ", data)
            #         ### Save Data to CSV ###
            #         self.csv_data['X'].append(self.ekf.state[0])
            #         self.csv_data['Y'].append(self.ekf.state[1])
            #         self.csv_data['Acc_x'].append(self.data["ACC"][0])
            #         self.csv_data['Acc_y'].append(self.data["ACC"][1])
            #         self.csv_data['heading'].append(self.data["ANGLE"][2])
            #         self.csv_data['transformed_heading'].append(wrap180(self.data["ANGLE"][2] + self.heading_offset))
            #         output_csv_path = './EKF_Data.csv'
            #         self.save_data_to_csv(output_csv_path)

                    # ############# Calculate Steering ####################
                    # try:
                    #     closest_point, next_target_points = find_closest_and_next_points((self.ekf.state[0], self.ekf.state[1]), self.ref_path, horizon=3)
                    #     angle = find_angle((self.ekf.state[0], self.ekf.state[1]), [closest_point, next_target_points[0]])
                    #     if angle < 120:
                    #         self.ref_path.remove(closest_point)
                    #         closest_point, next_target_points = find_closest_and_next_points((self.ekf.state[0], self.ekf.state[1]), self.ref_path, horizon=3)
                    #     ab = np.array([closest_point[0] - self.ekf.state[0], closest_point[1] - self.ekf.state[1]])
                    #     if np.linalg.norm(ab) < 15:
                    #         self.ref_path.remove(closest_point)
                    #         closest_point, next_target_points = find_closest_and_next_points((self.ekf.state[0], self.ekf.state[1]), self.ref_path, horizon=3)
                        
                    #     steering, tmp1, tmp2 = find_steering_angle((self.ekf.state[0], self.ekf.state[1]), closest_point, next_target_points[0], wrap180(self.data["ANGLE"][2] + self.heading_offset))
                    #     steering = int(np.clip(steering, -25, 25))
                    #     # print("Current Position:", self.ekf.state[0], self.ekf.state[1])
                    #     # print("Next Target 1:", closest_point)
                    #     # print("Next Target 2:", next_target_points[0])
                    #     # print("Steering: ",-steering)
                    #     # print("XAB:", tmp1, "XAC", tmp2)
                    #     # print("Heading:", wrap180(self.data["ANGLE"][2] - self.heading_offset))
                    #     self.control.setAngle(-steering)
                    # except:
                    #     # print("Out Range!!!!")
                    #     continue

            #         ################################################################################


    # =============================== START ===============================================
    def start(self):
        super(threadWitMotion, self).start()

        

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
)
from src.templates.threadwithstop import ThreadWithStop
import socket
import json
import math
import signal
from lib.IMU_GPS_EKF_5States import ExtendedKalmanFilter
from lib.Transform import Transform
import serial
import pynmea2
import struct

## MPC ##
from lib.MPC_kinematic_bycicle_model import KinematicBicycleModel

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
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, debugger):
        super(threadWitMotion, self).__init__()
        self.queuesList = queuesList
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
        self.csv_data = {
            "X": [],
            "Y": [],
            "Z": [],
            "quality": [],
            "heading": [],
            ### Raw ###
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
        self.ekf = ExtendedKalmanFilter()
        self.transform = Transform()
        self.initialize = False

        self.SERIAL_PORT = '/dev/ttyUSB0'        # Windows
        self.BAUDRATE = 230400  # thường GPS module là 9600
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=1)

        self.prev_time = 0.0
        self.prev_lat = 0.0
        self.prev_lon = 0.0
        self.gps_thres = 0.0
        ################### MPC #####################

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
    
    def calculate_gnss_heading(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        # Calculate the difference in longitude
        delta_lon = lon2 - lon1
        # Calculate the initial bearing
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        initial_bearing = math.atan2(x, y)
        # Convert bearing from radians to degrees
        initial_bearing = math.degrees(initial_bearing)
        # Normalize the bearing to 0-360 degrees
        heading = (initial_bearing + 360) % 360
        return heading
    
    def save_data_to_csv(self, csv_file):
        df = pd.DataFrame(self.csv_data)

        df.to_csv(csv_file, mode='a', index=False, header=False)

        for key in self.csv_data.keys():
            self.csv_data[key].clear()
    
    # =============================== MPC =================================================
    def normalize(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def find_closest_and_next_points(self, current_position, reference_path, horizon = 3):
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
            index = []
            for i in range(closest_index + 1, len(reference_path)):
                index.append(i)
            return closest_point, reference_path[closest_index + 1:], index
        else:
            index = []
            for i in range(closest_index + 1, closest_index+horizon+1):
                index.append(i)
            return closest_point, reference_path[closest_index + 1: closest_index+horizon+1], index

    def calculate_heading(self, x1, y1, x2, y2):
        """
        Calculate the heading (angle) between two points in 2D space.

        Parameters:
        - x1, y1: Coordinates of the first point.
        - x2, y2: Coordinates of the second point.

        Returns:
        - heading (float): Heading angle in degrees, where:
        - 0 degrees points to the positive x-axis,
        - 90 degrees points to the positive y-axis,
        - 180 degrees points to the negative x-axis,
        - -90 degrees points to the negative y-axis.
        """
        delta_x = x2 - x1
        delta_y = y2 - y1

        # Calculate the angle in radians
        heading_rad = math.atan2(delta_y, delta_x)

        return self.normalize(heading_rad)

    def convert_360to180(self, angle):
        return ((angle + 180) % 360) - 180
    
    def objective(self, controls, cur_state, closest_point, ref_path, index, dt):
        cost = 0
        weights = [0.1, 2.5, 0.1, 0.1, 0.1] # vel, steer, x, y, heading
        x, y, heading = cur_state
        for i in range(len(ref_path)):
            control_vel = controls[i]
            control_steering = controls[len(ref_path) + i]
            ref_x, ref_y = ref_path[i]
            cost_vel = weights[0] * control_vel ** 2
            cost_steer = weights[1] * control_steering ** 2
            cost_x = weights[2] * (ref_x - x) ** 2
            cost_y = weights[3] * (ref_y - y) ** 2
            # if i == 0:
            #     ref_heading = self.calculate_heading(closest_point[0], closest_point[1], ref_x, ref_y)
            # else:
            #     ref_heading = self.calculate_heading(ref_path[i-1][0], ref_path[i-1][1], ref_x, ref_y)
            ref_heading = self.ref_heading[index[i]]
            # print('X:', x, 'Y:', y)
            # print('Ref X:', ref_x, 'Ref Y:', ref_y)
            # print('Closest Point:', closest_point[0], closest_point[1])
            # print('Heading:', heading)
            # print('Ref Heading:', ref_heading)
            cost_heading = weights[4] * (ref_heading - heading) ** 2
            cost += (cost_vel + cost_steer + cost_x + cost_y + cost_heading) / np.sum(weights)
            x, y, heading, _, _, _ = self.kbm.discrete_kbm(control_vel, control_steering, x, y, heading, 0.0, dt)
        return cost
    
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
        while self._running:
            byte = self.ser.read()
            if byte:
                buffer += byte

                if buffer[0] != 0x55:
                    buffer.pop(0)
                    continue

                if len(buffer) >= 11:
                    sensor_type, values = self.parse_data(buffer[:11])
                    if sensor_type:
                        if sensor_type == 'ACC':
                            # print(f"Acceleration [g]: {values}")
                            self.data[sensor_type] = values
                            self.full = False
                        elif sensor_type == 'GYRO':
                            # print(f"Gyroscope [°/s]: {values}")
                            self.data[sensor_type] = values
                            self.full = False
                        elif sensor_type == 'ANGLE':
                            # print(f"Euler angles [°]: {values}")
                            self.data[sensor_type] = values
                            self.full = False
                        elif sensor_type == 'TIME':
                            year, month, day, hour, minute, second, ms = values
                            self.data[sensor_type] = float(hour) * 3600 + float(minute) * 60 + float(second) + float(ms)/1000
                            self.full = False
                        else:
                            self.full = True
                    buffer = buffer[11:]
            if self.full:
                gps = {"x": -1.0, "y": -1.0, "z": 0.0, "quality": 0}
                if not self.queuesList["CarGPSInfo"].empty():
                    gps = self.queuesList["CarGPSInfo"].get()['msgValue']
                    if gps["quality"] < self.gps_thres:
                        gps = {"x": -1.0, "y": -1.0, "z": 0.0, "quality": 0}
                ### Save Raw Data to CSV ###
                self.csv_data['X'].append(gps["x"])
                self.csv_data['Y'].append(gps["y"])
                self.csv_data["Z"].append(gps["z"])
                self.csv_data["quality"].append(gps["quality"])
                self.csv_data['heading'].append(self.data["ANGLE"][2])

                self.csv_data["Acc_x"].append(self.data["ACC"][0])
                self.csv_data["Acc_y"].append(self.data["ACC"][1])
                self.csv_data["Acc_z"].append(self.data["ACC"][2])
                self.csv_data["Gyro_x"].append(self.data["GYRO"][0])
                self.csv_data["Gyro_y"].append(self.data["GYRO"][1])
                self.csv_data["Gyro_z"].append(self.data["GYRO"][2])
                self.csv_data["Angle_x"].append(self.data["ANGLE"][0])
                self.csv_data["Angle_y"].append(self.data["ANGLE"][1])
                self.csv_data["Angle_z"].append(self.data["ANGLE"][2])
                self.csv_data["Time"].append(self.data["TIME"])

                print("GPS:", gps)
                # print("Data:", self.data)

                output_csv_path = './Raw_Data.csv'
                self.save_data_to_csv(output_csv_path)
                # if not self.initialize:
                #     if gps["x"] != -1.0 and gps["y"] != -1.0:
                #         self.initialize = True
                #         self.prev_lat = gps["x"]
                #         self.prev_lon = gps["y"]
                #         self.ekf.state = [gps["x"], gps["y"], self.data["ANGLE"][2], 0.0, 0.0]
                #         self.prev_time = self.data["TIME"]
                # else:
                #     dt = self.data["TIME"] - self.prev_time
                #     self.prev_time = self.data["TIME"]
                #     if (self.prev_lat != gps["x"] and self.prev_lon != gps["y"]) and (gps["x"] != -1.0 and gps["y"] != -1.0):
                #         self.ekf.predict(self.data["ACC"][0], self.data["ACC"][1], self.data["ANGLE"][2], dt)
                #         self.ekf.update_gps(gps["x"], gps["y"])
                #         self.prev_lat = gps["x"]
                #         self.prev_lon = gps["y"]
                #     else:
                #         self.ekf.predict(self.data["ACC"][0], self.data["ACC"][1], self.data["ANGLE"][2], dt)
                #     if dt != 0:
                #         print("Predict: ", self.ekf.state)
                #         data = {
                #             "x": self.ekf.state[0],
                #             "y": self.ekf.state[1],
                #             "z": gps["z"],
                #             "quality": gps["quality"],
                #             "type": "location",
                #             "id": 3,
                #         }
                #         self.queuesList[Position.Queue.value].put(
                #         {
                #             "Owner": Position.Owner.value,
                #             "msgID": Position.msgID.value,
                #             "msgType": Position.msgType.value,
                #             "msgValue": data,
                #         })
                #     # if self.count % 42 == 0:
                #     #     self.queuesList[Position.Queue.value].put(
                #     #     {
                #     #         "Owner": Position.Owner.value,
                #     #         "msgID": Position.msgID.value,
                #     #         "msgType": Position.msgType.value,
                #     #         "msgValue": data,
                #     #     })
                #     #     self.count = 0
                #     # self.count += 1

                #     ### Save Raw Data to CSV ###
                #     self.csv_data['X'].append(gps["x"])
                #     self.csv_data['Y'].append(gps["y"])
                #     self.csv_data['heading'].append(self.data["ANGLE"][2])

                #     # self.csv_data["Acc_x"].append(self.data["ACC"][0])
                #     # self.csv_data["Acc_y"].append(self.data["ACC"][1])
                #     # self.csv_data["Acc_z"].append(self.data["ACC"][2])
                #     # self.csv_data["Gyro_x"].append(self.data["GYRO"][0])
                #     # self.csv_data["Gyro_y"].append(self.data["GYRO"][1])
                #     # self.csv_data["Gyro_z"].append(self.data["GYRO"][2])
                #     # self.csv_data["Angle_x"].append(self.data["ANGLE"][0])
                #     # self.csv_data["Angle_y"].append(self.data["ANGLE"][1])
                #     # self.csv_data["Angle_z"].append(self.data["ANGLE"][2])
                #     # self.csv_data["Time"].append(self.data["TIME"])

                #     output_csv_path = './Raw_Data.csv'
                #     self.save_data_to_csv(output_csv_path)
                #     ### Save Data to CSV ###
                #     # self.csv_data['X'].append(self.ekf.state[0])
                #     # self.csv_data['Y'].append(self.ekf.state[1])
                #     # self.csv_data['heading'].append(self.data["ANGLE"][2])
                #     # output_csv_path = './EKF_Data.csv'
                #     # self.save_data_to_csv(output_csv_path)


    # =============================== START ===============================================
    def start(self):
        super(threadWitMotion, self).start()

        

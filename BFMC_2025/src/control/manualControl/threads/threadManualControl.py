# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import cv2
import threading
import base64
import time
import numpy as np
import os
import sys
import curses
import pandas as pd

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Recording,
    Record,
    Config,
    SpeedMotor,
    SteerMotor,
    Speed,
    Steer,
    Position,
)
from src.templates.threadwithstop import ThreadWithStop

from src.utils.CarControl.CarControl import CarControl

import Jetson.GPIO as GPIO

class threadManualControl(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, Speed, Steer, Reset, debugger):
        super(threadManualControl, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.debugger = debugger
        self.frame_rate = 5
        self.recording = False
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.video_writer = ""
        self.subscribe()
        self.Configs()
        self.stdscr = curses.initscr()
        curses.cbreak()
        self.stdscr.keypad(1)
        self.speed = 0
        self.angle = 0
        self.Speed, self.Steer = Speed, Steer
        self.Reset = Reset
        # pipeRecvIMU, pipeSendIMU = Pipe(duplex=False)
        # self.pipeRecvIMU = pipeRecvIMU
        # self.pipeSendIMU = pipeSendIMU
        # pipeRecvVLX, pipeSendVLX = Pipe(duplex=False)
        # self.pipeRecvVLX = pipeRecvVLX
        # self.pipeSendVLX = pipeSendVLX
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(15, GPIO.OUT)
        GPIO.setup(31, GPIO.OUT)
        GPIO.output(31, GPIO.LOW)
        GPIO.output(15, GPIO.HIGH)
        print('Initialize manual control thread!!!')

        self.csv_data = {
            "X": [],
            "Y": [],
            "Z": [],
            "Conf": [],
        }
        self.count = 0
        GPIO.output(15, GPIO.LOW)
        print("HARD RESET STM32!!!")
        time.sleep(5)
        GPIO.output(15, GPIO.HIGH)
        self.Reset.value = True
        time.sleep(2)
        self.Reset.value = False

    def Queue_Sending(self):
        self.control.setSpeed(self.speed)
        self.control.setAngle(self.angle)

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadManualControl", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadManualControl", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        # cv2.destroyAllWindows()
        self.speed = 0
        self.angle = 0
        self.Queue_Sending()
        super(threadManualControl, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    def save_data_to_csv(self, csv_file):
        df = pd.DataFrame(self.csv_data)

        df.to_csv(csv_file, mode='a', index=False, header=False)

        for key in self.csv_data.keys():
            self.csv_data[key].clear()

    def save_coords(self, key):
        while self.count < 100:
            gps = {"x": -1.0, "y": -1.0, "z": 0.0, "quality": 0}
            if not self.queuesList["CarGPSInfo"].empty():
                gps = self.queuesList["CarGPSInfo"].get()['msgValue']

            if (gps["x"] != -1.0 and gps["y"] != -1.0):
                data = {
                    "x": gps["x"],
                    "y": gps["y"],
                    "z": gps["z"],
                    "quality": gps["quality"],
                    "heading": self.data["ANGLE"][2],
                    "type": "location",
                    "id": 3,
                }
                self.csv_data["X"].append(data["x"])
                self.csv_data["Y"].append(data["y"])
                self.csv_data["Z"].append(data["z"])
                self.csv_data["Conf"].append(data["quality"])
                output_csv_path = f'./point{key}.csv'
                self.save_data_to_csv(output_csv_path)
                self.count += 1
        self.count = 0
        print(f"Complete Point {key}!!!!")
        


    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            # print(self.control.getIMUdata())
            key = self.stdscr.getch()
            # Speed
            if key & 0xFF == ord('r'):
                GPIO.output(15, GPIO.LOW)
                print("RESET STM32!!!")
                time.sleep(0.5)
                GPIO.output(15, GPIO.HIGH)
                self.Reset.value = True
                time.sleep(2)
                self.Reset.value = False

            if  key & 0xFF == ord('q'):
                self.speed = 0
                self.Queue_Sending()
            elif key & 0xFF == ord('w'):
                if self.speed < 30:
                    self.speed += 5
                
                self.Queue_Sending()
            elif key & 0xFF == ord('s'):
                if self.speed > -30:
                    self.speed -= 5
                    
                self.Queue_Sending()

            
            # Steer
            if  key & 0xFF == ord('e'):
                self.angle = 0
                self.Queue_Sending()
            elif key & 0xFF == ord('d'):
                # if self.angle < 30:
                #     self.angle += 1
                self.angle = 20
                self.Queue_Sending()
            elif key & 0xFF == ord('a'):
                # if self.angle > -30:
                #     self.angle -= 1
                self.angle = -20
                self.Queue_Sending()
            # print(f'Angle: {self.angle}')
            # print(f'Speed: {self.speed}')
            elif key & 0xFF == ord('t'):
                self.control.enVLX(200)
            elif key & 0xFF == ord('y'):
                print(self.control.getVLXdata())

            # Forward
            if key & 0xFF == ord('c'):
                print("Forward")
                self.angle = 0
                self.speed = 25
                self.Queue_Sending()
                time.sleep(5.6)
                self.angle = 0
                self.speed = 0
                self.Queue_Sending()
                # time.sleep(1)
            # Curve Left
            if key & 0xFF == ord('v'):
                print("Curve Left")
                self.angle = -7
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 0
                self.speed = 25
                self.Queue_Sending()
                time.sleep(3)
                self.angle = -24
                self.speed = 25
                self.Queue_Sending()
                time.sleep(2.2)
                self.angle = -14
                self.speed = 25
                self.Queue_Sending()
                time.sleep(1)
                self.angle = 0
                self.speed = 0
                self.Queue_Sending()
                # time.sleep(2.5)
            # Curve Right
            if key & 0xFF == ord('b'):
                print("Curve Right")
                self.angle = 7
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 14
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 17
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 24
                self.speed = 25
                self.Queue_Sending()
                time.sleep(3)
                self.angle = 0
                self.speed = 0
                self.Queue_Sending()
                # time.sleep(1)
            # Roundabout Right
            if key & 0xFF == ord('n'):
                print("Roundabout Right")
                self.angle = 5
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 15
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 24
                self.speed = 25
                self.Queue_Sending()
                time.sleep(2)
                self.angle = 0
                self.speed = 25
                self.Queue_Sending()
                time.sleep(2)
                self.angle = 5
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 15
                self.speed = 0
                self.Queue_Sending()
                time.sleep(0.1)
                self.angle = 24
                self.speed = 25
                self.Queue_Sending()
                time.sleep(2)
                self.angle = 0
                self.speed = 0
                self.Queue_Sending()
                # time.sleep(1)
            # Roundabout Forward
            # if key & 0xFF == ord('m'):
            #     print("Roundabout Forward")
            #     self.angle = 0
            #     self.speed = 0
            #     self.Queue_Sending()
            #     time.sleep(0.1)
            #     self.angle = 0
            #     self.speed = 25
            #     self.Queue_Sending()
            #     time.sleep(4)
            #     self.angle = 0
            #     self.speed = 0
            #     self.Queue_Sending()
                # time.sleep(1)
    # =============================== START ===============================================
    def start(self):
        super(threadManualControl, self).start()

        

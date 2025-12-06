import cv2
import threading
import base64
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Segmentation,
    Record,
    Config,
    ObjectDetection,
    Points
)
from src.templates.threadwithstop import ThreadWithStop
from src.decisionMaking.PathPlanning import PathPlanning

import torch
import math
import json

from src.utils.CarControl.CarControl import CarControl

class threadDecisionMaking(ThreadWithStop):
    
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, Speed, Steer, debugger):
        super(threadDecisionMaking, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.speed = 0
        self.angle = 0
        self.Speed, self.Steer = Speed, Steer
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        
        self.debugger = debugger
        self.subscribe()
        self.Configs()

        self.direction_picking = PathPlanning(config_file='config/output.graphml')
    
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
                "To": {"receiver": "threadDecisionMaking", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadDecisionMaking", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        # cv2.destroyAllWindows()
        super(threadDecisionMaking, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    def get_sensor_data(self):
        
        try:
            msgValue = self.control.getVLXdata()['msgValue']
        except:
            msgValue = self.control.getVLXdata()
            print("Error in getting VLX data")
        return msgValue

    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        # time.sleep(15)
        # self.control.enVLX(200)
            
        while self._running:
            self.control.enVLX(200)
            print(self.get_sensor_data())
            time.sleep(1)
            
    
     # =============================== START ===============================================
    def start(self):
        super(threadDecisionMaking, self).start()
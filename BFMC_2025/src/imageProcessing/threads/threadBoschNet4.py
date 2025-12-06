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
    Points,
    DecisionMaking,
    AngleLaneKeeping,
    TrafficLog,
)
from src.templates.threadwithstop import ThreadWithStop
from src.imageProcessing.threads.infer_trt import TRT
import torch
import math
from src.imageProcessing.InferenceBoschNet import InferenceBoschNet
from lib.LaneKeeping4 import LaneKeeping
from src.utils.CarControl.CarControl import CarControl

import Jetson.GPIO as GPIO

class threadBoschNet(ThreadWithStop):
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, Speed, Steer, Reset, debugger):
        super(threadBoschNet, self).__init__()
        self.queuesList = queuesList
        self.Reset = Reset
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        
        self.debugger = debugger
        self.subscribe()
        self.Configs()
        
        self.lanekeeping = LaneKeeping()
        self.Beta = 0.5
        self.Speed, self.Steer = Speed, Steer
        self.speed, self.angle = 0, 0
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        
        self.width = 640.0
        self.height = 480.0
        self.fps = 30
        self.inference_boschnet = InferenceBoschNet(model_file='./models/model_14.engine', debugger=debugger)
        
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(15, GPIO.OUT)
        GPIO.setup(31, GPIO.OUT)
        GPIO.output(31, GPIO.LOW)
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(15, GPIO.LOW)
        print("HARD RESET STM32!!!")
        time.sleep(3)
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
                "To": {"receiver": "threadBoschNet", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadBoschNet", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        # cv2.destroyAllWindows()
        super(threadBoschNet, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            # print(message)
        threading.Timer(1, self.Configs).start()
    
    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            if not self.queuesList["BoschNetCamera"].empty():
                # start = time.time()
                request = self.queuesList["BoschNetCamera"].get()["msgValue"]
                image_data = base64.b64decode(request)
                request = np.frombuffer(image_data, dtype=np.uint8)     
                request = cv2.imdecode(request, cv2.IMREAD_COLOR)
                # print(request.shape)
                
                obj_msg = {}
                lane_img, output_obj =  self.inference_boschnet.inference(request)
                img_obj, classes, areas = output_obj
                # is_lane is flag return when lanekeeping can do the job or not
                self.speed, self.angle, drawn_lane_img, is_lane = self.lanekeeping.AngCal(lane_img)
                self.angle -= self.Beta * (self.angle - self.angle)
                self.angle = int(self.angle + 0.5)
                # self.Queue_Sending()
                # print(lane_img.shape)
                _, encoded_img = cv2.imencode(".jpg", lane_img)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                
                self.queuesList[Segmentation.Queue.value].put(
                    {
                        "Owner": Segmentation.Owner.value,
                        "msgID": Segmentation.msgID.value,
                        "msgType": Segmentation.msgType.value,
                        "msgValue": image_data_encoded,
                    }
                )
                _, encoded_img = cv2.imencode(".jpg", img_obj)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                obj_msg = {"Image": image_data_encoded, "Class": classes, "Area": areas}
                self.queuesList[ObjectDetection.Queue.value].put(
                    {
                        "Owner": ObjectDetection.Owner.value,
                        "msgID": ObjectDetection.msgID.value,
                        "msgType": ObjectDetection.msgType.value,
                        "msgValue": obj_msg,
                    }
                )
                # print(obj_msg)
                self.queuesList[TrafficLog.Queue.value].put(
                    {
                        "Owner": TrafficLog.Owner.value,
                        "msgID": TrafficLog.msgID.value,
                        "msgType": TrafficLog.msgType.value,
                        "msgValue": classes,
                    }
                )
                
                self.queuesList[DecisionMaking.Queue.value].put(
                    {
                        "Owner": DecisionMaking.Owner.value,
                        "msgID": DecisionMaking.msgID.value,
                        "msgType": DecisionMaking.msgType.value,
                        "msgValue": {"Class": obj_msg["Class"], "Area": obj_msg["Area"]},
                    }
                )

                _, encoded_img = cv2.imencode(".jpg", drawn_lane_img)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                
                self.queuesList[Points.Queue.value].put(
                    {
                        "Owner": Points.Owner.value,
                        "msgID": Points.msgID.value,
                        "msgType": Points.msgType.value,
                        "msgValue": image_data_encoded,
                    }
                )
                self.queuesList[AngleLaneKeeping.Queue.value].put(
                    {
                        "Owner": AngleLaneKeeping.Owner.value,
                        "msgID": AngleLaneKeeping.msgID.value,
                        "msgType": AngleLaneKeeping.msgType.value,
                        "msgValue": {"angle": self.angle, "is_lane": is_lane},
                    }
                )
                
                # print("FPS: ", 1/(time.time()-start))
                # print(self.speed)
                
                
    def start(self):
        super(threadBoschNet, self).start()
    
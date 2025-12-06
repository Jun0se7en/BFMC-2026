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
from decisionMaking.oldPathPlanning import DirectionPicking

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
        
        self.debugger = debugger
        self.subscribe()
        self.Configs()
        
        self.Beta = 0.5
        self.Speed, self.Steer = Speed, Steer
        self.speed, self.angle = 0, 0
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        
        intersection_cfg = 'main_rc.json'
        event_cfg = 'signs_area.json'
        nodes_cfg = 'map_nodes.json'
        self.msg = {}
        self.intersection_detection = IntersectionDetection(config_file=intersection_cfg)
        self.direction_picking = DirectionPicking(config_file=nodes_cfg)
        self.car_instruction = CarInstruction()
        self.sign_config = self.load_config_file(event_cfg)
        self.sign_actions = self.car_instruction.sign_actions
        
        self.recorded_time = 0
        self.trusted_sign = False
        
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
    
    @staticmethod
    def load_config_file(config_file):
        with open(config_file, "r") as jsonfile:
            data = json.load(jsonfile)
        return data
    
    def get_dict(self, obj_msg):
        for key, value in self.sign_actions.items():
            if value.get("obj_msg") == obj_msg:
                # return name of function and area threshold
                return key, value.get("area")
        return None
    
    def get_sign_attr(self, filtered_obj_msg):
        obj_class, curr_area = filtered_obj_msg 
        result = self.get_dict(obj_class)

        if result is None:
            self.recorded_time = 0
            return None

        sign_msg, thresh_area = result

        if curr_area > thresh_area:
            if self.recorded_time == 0:
                self.recorded_time = time.time()
                return None
            if time.time() - self.recorded_time > 0.7:
                instruction_set = self.sign_actions.get(sign_msg, lambda: None)()
                self.recorded_time = 0
                return instruction_set
        else:
            self.recorded_time = 0

        return None
    
    def exec_instruction(self, instruction_set):
        for speed_angle, time_sec in instruction_set:
            speed, angle = speed_angle
            if all(value is not None for value in (speed, angle, time_sec)):   
                self.speed = speed
                self.angle = angle
                self.control.setControl(self.speed, self.angle, 1)
                time.sleep(time_sec)
                
    def obj_filter(self, obj_msg):
        obj_lst = obj_msg.get("Class", [])
        area_lst = obj_msg.get("Area", [])
        
        if not area_lst or not obj_lst or len(obj_lst) != len(area_lst):
            return None
        
        max_index = area_lst.index(max(area_lst))
        curr_area = area_lst[max_index]
        obj_class = obj_lst[max_index]
        
        return (obj_class, curr_area)
    
    # Call this, output is a set of instruction affect immediately to the engine
    def recieve_msg(self, obj_msg):
        filtered_obj_msg = self.obj_filter(obj_msg)     
        if filtered_obj_msg is not None:  
            instruction_set = self.get_sign_attr(filtered_obj_msg)
            if instruction_set is not None:
                self.exec_instruction(instruction_set)
    
    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            
            
            if not self.queuesList["DecisionMaking"].empty():
                self.msg = self.queuesList["DecisionMaking"].get()["msgValue"]
                # print(self.msg["Area"])
            # start = time.time()
            # obj_msg = self.obj_filter(msg)
            # if obj_msg is not None:  
            #     instruction_set = self.get_sign_attr(obj_msg)
            #     if instruction_set is not None:
            #         self.exec_instruction(instruction_set)
            # intersection_msg = self.IntersectionDetection.intersection_check(img, False)
            # print(intersection_msg)
            # print("FPS: ", 1/(time.time()-start))
    
     # =============================== START ===============================================
    def start(self):
        super(threadDecisionMaking, self).start()
            
import cv2
import threading
import base64
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from collections import Counter
from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Record,
    Config,
)
from src.templates.threadwithstop import ThreadWithStop
from src.utils.CarControl.CarControl import CarControl

from src.decisionMaking.planning.PathFinder import PathFinder
from src.decisionMaking.logic.logicHandler import LogicHandler
from src.decisionMaking.logic.nodeHandler import NodeHandler
from src.decisionMaking.planning.Navigator import Navigator
from src.decisionMaking.logic.randomStartHandler import RandomStartHandler
from src.decisionMaking.logic.Perception import Perception
import json

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

        self.Speed, self.Steer = Speed, Steer
        self.speed, self.angle = 0, 0
        self.new_speed = 0
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)
        
        graph_path = "src/decisionMaking/cfg/output.graphml"
        cfg_path = "src/decisionMaking/cfg/cfg.json"
        area_path = "src/decisionMaking/cfg/sign_area.json"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(threadName)s] %(message)s",
            handlers=[logging.StreamHandler()],
        )
        
        self.pth_finder = PathFinder(graph_path)
        self.handler = NodeHandler(cfg_path, graph_path)
        self.random_start_handler = RandomStartHandler(graph_path)
        self.nav = Navigator(graph_path, cfg_path)
        self.perception = Perception(area_path=area_path)
        
        # print(list(self.nav.intersections))
    
    def Queue_Sending(self):
        # print(self.speed)
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
            # print(message)
        threading.Timer(1, self.Configs).start()
    
    # ================================ RUN ================================================
    def run(self):
        with open('src/decisionMaking/cfg/sign_area.json', 'r') as f:
            area_cfg = json.load(f)
        area_threshold = {k.lower(): v["area"] for k, v in area_cfg.items()}

        object_keys = [
            "stop", "crosswalk", "redlight", "yellowlight", "greenlight",
            "highwayentry", "highwayend", "car", "pedestrian",
            "parking", "roundabout", "noentry", "oneway", "priority"
        ]
        lane_angle = 0
        # x_pos = 1.94
        # y_pos = 0.36
        head_pos = 0
        x_pos = 0
        y_pos = 0
        init_heading = 0
        while x_pos == 0 or y_pos == 0:
            if not self.queuesList["CarGPSInfo"].empty():
                msg = self.queuesList["CarGPSInfo"].get()["msgValue"]
                x_pos = msg["x"]
                y_pos = msg["y"]
            # print(x_pos, y_pos)
        if not self.queuesList["Heading"].empty():
            msg = self.queuesList["Heading"].get()["msgValue"]
            init_heading = int(msg)
        gps_data = [
        (x_pos, y_pos, init_heading)        # (2.87, 2.26, init_heading),
        # (2.86, 2.27, init_heading),
        # (2.87, 2.28, 85)
        ]
        prev_node, current_node = self.random_start_handler.find_best_matching_edge(gps_data)
        print('Init current node: ', current_node)
        # current_node = 264
        self.nav.capture_all(current_node)
        print("Direction: ", self.nav.get_direction_map())
        # print(self.nav.full_path)
        curr_x, curr_y = 0, 0
        while self._running:
            if not self.queuesList["CarGPSInfo"].empty():
                msg = self.queuesList["CarGPSInfo"].get()["msgValue"]
                x_pos = msg["x"]
                y_pos = msg["y"]
            if curr_x != x_pos:
                curr_x = x_pos
            if curr_y != y_pos:
                curr_y = y_pos
            if not self.queuesList["Heading"].empty():
                msg = self.queuesList["Heading"].get()["msgValue"]
                head_pos = int(msg)

            gps_sample = [(curr_x, curr_y, head_pos)]
            # print(gps_sample)
            
            prev_node, current_node = self.random_start_handler.find_best_matching_edge(gps_sample)
            # print(current_node, gps_sample)
            current_node = current_node
            next_node = int(self.handler.get_successors(current_node)[0])
            if not next_node:
                continue
            if next_node in self.nav.intersections:
                direction = self.nav.get_direction_map()[next_node]
                # direction = "right"
                self.perception.handler.set_direction(direction)
                print("Next intersection: ", next_node, direction)
                
            if current_node in self.nav.intersections:
                for segment in self.nav.full_path:
                    if current_node in segment:
                        idx = segment.index(current_node)
                        if idx + 1 < len(segment):
                            expected_next_node = segment[idx + 1]
                            if next_node != expected_next_node:
                                self.nav.capture_all(next_node)
                        break

            if not self.queuesList["AngleLaneKeeping"].empty():
                lane_angle = self.queuesList["AngleLaneKeeping"].get()["msgValue"]['angle']
                is_lane = self.queuesList["AngleLaneKeeping"].get()["msgValue"]['is_lane']
                self.perception.update_angle(lane_angle)
                area_sample = {
                "intersection": is_lane
                }
                # print("Lane angle: ", lane_angle)
                self.perception.update(**area_sample)

            if not self.queuesList["DecisionMaking"].empty():
                request = self.queuesList["DecisionMaking"].get()["msgValue"]
                # Initialize sample dict
                raw_sample = {key: False for key in object_keys}
                raw_sample.update({f"{key}_area": 0 for key in object_keys})

                if len(request['Class']) != 0:
                    # print(request)
                    for cls, area in zip(request['Class'], request['Area']):
                        cls_key = cls.lower()
                        if cls_key in object_keys:
                            threshold = area_threshold.get(cls_key, 0)
                            if area > threshold:
                                raw_sample[cls_key] = True

                # print(raw_sample)
                self.perception.update(**raw_sample)

            self.angle = self.perception.get_current_angle()
            # print(self.angle)
            # self.angle = int(lane_angle)
            self.speed = self.perception.get_current_speed()
            # if self.speed != 0:
            # print(self.angle)
            # print(self.perception.get_current_angle(), lane_angle)
            self.Queue_Sending()

                      
            

    
     # =============================== START ===============================================
    def start(self):
        super(threadDecisionMaking, self).start()
            
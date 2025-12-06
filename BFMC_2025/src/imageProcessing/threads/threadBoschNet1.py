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
)
from src.templates.threadwithstop import ThreadWithStop
from src.imageProcessing.threads.infer_trt import TRT
import torch
import math
from src.utils.utils import non_max_suppression, scale_coords, plot_one_box
from lib.LaneKeeping5 import LaneKeeping
# import tensorflow as tf
from src.utils.CarControl.CarControl import CarControl

import Jetson.GPIO as GPIO

# Use this thread for LaneLine Segmentation
class threadBoschNet(ThreadWithStop):
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
        super(threadBoschNet, self).__init__()
        self.queuesList = queuesList
        self.Reset = Reset
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.model = TRT('./models/model_4.engine')
        self.W_ = 640
        self.H_ = 384
        self.debugger = debugger
        self.subscribe()
        self.Configs()
        self.names = {
            0: 'Car',
            1: 'CrossWalk',
            2: 'Greenlight',
            3: 'HighwayEnd',
            4: 'HighwayEntry',
            5: 'NoEntry',
            6: 'OneWay',
            7: 'Parking',
            8: 'Pedestrian',
            9: 'PriorityRoad',
            10: 'Redlight',
            11: 'Roundabout',
            12: 'Stop',
            13: 'Yellowlight',
        }
        self.colors = [
            (255, 0, 0),      # Car - Red
            (0, 255, 0),      # CrossWalk - Green
            (0, 0, 255),      # Greenlight - Blue
            (255, 255, 0),    # HighwayEnd - Yellow
            (255, 165, 0),    # HighwayEntry - Orange
            (128, 0, 128),    # NoEntry - Purple
            (0, 255, 255),    # OneWay - Cyan
            (255, 192, 203),  # Parking - Pink
            (139, 69, 19),    # Pedestrian - Brown
            (128, 128, 0),    # PriorityRoad - Olive
            (255, 69, 0),     # Redlight - Red-Orange
            (0, 128, 128),    # Roundabout - Teal
            (169, 169, 169),  # Stop - DarkGray
            (255, 215, 0)     # Yellowlight - Gold
        ]
        self.lanekeeping = LaneKeeping()
        self.Beta = 0.5
        self.Speed, self.Steer = Speed, Steer
        self.speed, self.angle = 0, 0
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)

        # self.width = 1280.0
        # self.height = 720.0
        # self.fps = 90
        self.width = 640.0
        self.height = 480.0
        self.fps = 30
        # self._init_camera()
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(15, GPIO.OUT)
        GPIO.setup(31, GPIO.OUT)
        GPIO.output(31, GPIO.LOW)
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(15, GPIO.LOW)
        print("HARD RESET STM32!!!")
        time.sleep(2)
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
            print(message)
        threading.Timer(1, self.Configs).start()
    
    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:

            if not self.queuesList["BoschNetCamera"].empty():
                start = time.time()
                request = self.queuesList["BoschNetCamera"].get()["msgValue"]
                image_data = base64.b64decode(request)
                request = np.frombuffer(image_data, dtype=np.uint8)     
                request = cv2.imdecode(request, cv2.IMREAD_COLOR)

                img = cv2.resize(request, (self.W_, self.H_))
                ### 
                img_obj = img.copy()
                img_rs = np.zeros((self.H_, self.W_, 3))

                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img= torch.from_numpy(img)
                img = torch.unsqueeze(img, 0)  # add a batch dimension
                img=img.cuda().float() / 255.0
                img = img.cuda()
                print('running ')
                img_out = self.model(img)

                inf_out = img_out[1]
                # print("Before NMS")
                det_pred = non_max_suppression(inf_out, conf_thres=0.7, iou_thres=0.7, classes=None, agnostic=True)
                # print("After NMS")
                det=det_pred[0]

                x0=img_out[0]

                _,da_predict=torch.max(x0, 1)

                DA = da_predict.byte().cpu().data.numpy()[0]*255
                
                img_rs[DA>100]=[255, 255, 255]
                if img_rs.dtype == np.float64:  
                    img_rs = cv2.convertScaleAbs(img_rs) 
                img_rs = cv2.resize(img_rs, (320, 240))
                img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
                contours, _ = cv2.findContours(img_rs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                black_image=np.zeros((240,320,3),np.uint8)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(black_image, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((3, 3), np.uint8)
                black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel)
                black_image = cv2.erode(black_image, kernel, iterations = 3)
                black_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
                img_rs = black_image
                # img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
                # print(det.shape)
                classes = []
                areas = []
                obj_msg = {}
                if len(det) and self.debugger:
                    det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_obj.shape).round()
                    for *xyxy,conf,cls in reversed(det):
                        label_det_pred = f'{self.names[int(cls)]} {conf:.2f}'
                        # print(label_det_pred)
                        classes.append(self.names[int(cls)])
                        x_min, y_min, x_max, y_max = xyxy
                        area = (x_max - x_min) * (y_max - y_min)
                        area = area.item()
                        areas.append(area)
                        plot_one_box(xyxy, img_obj , label=label_det_pred, color=self.colors[int(cls)], line_thickness=2)
                    _, encoded_img = cv2.imencode(".jpg", img_obj)
                    image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                    obj_msg = {"Image": image_data_encoded, "Class": classes, "Area": areas}
                else:
                    det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_obj.shape).round()
                    for *xyxy,conf,cls in reversed(det):
                        classes.append(self.names[int(cls)])
                        x_min, y_min, x_max, y_max = xyxy
                        area = (x_max - x_min) * (y_max - y_min)
                        area = area.item()
                        areas.append(area)
                    _, encoded_img = cv2.imencode(".jpg", img_obj)
                    image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                    obj_msg = {"Image": image_data_encoded, "Class": classes, "Area": areas}

                #### Lane Keeping ####
                self.speed, self.angle, img = self.lanekeeping.AngCal(img_rs)
                
                # intersection_msg = self.IntersectionDetection.intersection_check(intersect_img, self.angle, False)
                # if intersection_msg == 'intersection':
                #     print(intersection_msg)
                #     self.speed = 0
                #     self.angle = 0
                #     self.Queue_Sending()
                #     time.sleep(5)
                # else:
                #     print(intersection_msg, self.angle)
                
                # left, right, img = self.lanekeeping.find_left_right_points(img_rs)
                # _, img = self.lanekeeping.find_middle_points_optimized(left, right, 240, 320, img)
                self.angle -= self.Beta * (self.angle - self.angle)
                self.angle = int(self.angle + 0.5)
                self.Queue_Sending()
                
                _, encoded_img = cv2.imencode(".jpg", img_rs)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                # if not self.queuesList[Segmentation.Queue.value].empty():
                #     _ = self.queuesList[Segmentation.Queue.value].get()
                # if not self.queuesList[ObjectDetection.Queue.value].empty():
                #     _ = self.queuesList[ObjectDetection.Queue.value].get()
                # if not self.queuesList[DecisionMaking.Queue.value].empty():
                #     _ = self.queuesList[DecisionMaking.Queue.value].get()
                # if not self.queuesList[Points.Queue.value].empty():
                #     _ = self.queuesList[Points.Queue.value].get()
                self.queuesList[Segmentation.Queue.value].put(
                    {
                        "Owner": Segmentation.Owner.value,
                        "msgID": Segmentation.msgID.value,
                        "msgType": Segmentation.msgType.value,
                        "msgValue": image_data_encoded,
                    }
                )
                self.queuesList[ObjectDetection.Queue.value].put(
                    {
                        "Owner": ObjectDetection.Owner.value,
                        "msgID": ObjectDetection.msgID.value,
                        "msgType": ObjectDetection.msgType.value,
                        "msgValue": obj_msg,
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

                _, encoded_img = cv2.imencode(".jpg", img)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                self.queuesList[Points.Queue.value].put(
                    {
                        "Owner": Points.Owner.value,
                        "msgID": Points.msgID.value,
                        "msgType": Points.msgType.value,
                        "msgValue": image_data_encoded,
                    }
                )

                # if self.debugger:
                print("FPS: ", 1/(time.time()-start))
                    # print("Speed: ", self.speed, "Angle: ", self.angle)
                    # print(obj_msg)

    # =============================== START ===============================================
    def start(self):
        super(threadBoschNet, self).start()
    
    def _init_camera(self):
        # self.camera = cv2.VideoCapture('./test.mp4')
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.camera.isOpened():
            print("Capture failed")
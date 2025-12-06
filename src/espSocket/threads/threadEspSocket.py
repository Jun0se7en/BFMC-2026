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

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    Record,
    Config,
    Segmentation,
    Points,
    CarControl,
    CarControlInfo,
    EKFCarControlInfo,
    RetrievingInfo,
)
from src.templates.threadwithstop import ThreadWithStop
import socket
import json
import math
import signal

DATATYPES = {
    "GET_ALL_DATA": 0,
    "GET_GPS": 1,
    "GET_SPEED_STEER": 2,
    "GET_IMU": 3,
    "GET_OBSTACLE": 4,
    "CMD_SPEED_STEER": 5,
    "REQ_SPEED_STEER": 6,
    "REQ_GPS": 7,
    "REQ_ALL_DATA": 8,
}

class TimeoutException(Exception):
    pass

def run_with_timeout(func, args=(), kwargs={}, timeout=5):
    """
    Runs a function with a timeout.
    :param func: Function to run
    :param args: Arguments for the function
    :param kwargs: Keyword arguments for the function
    :param timeout: Timeout in seconds
    :return: Function result if completed within the timeout, otherwise None
    """
    result = [None]
    exception = [None]

    if args != ():
        url = args
    else:
        url = None

    def target():
        try:
            if url is not None:
                result[0] = func(url)
            else:
                result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Function timed out after {timeout} seconds.")
        # Raise the timeout exception in the main thread
        raise TimeoutException()

    if exception[0]:
        # Re-raise the exception caught in the thread
        raise exception[0]

    return result[0]

class threadEspSocket(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, ip_address, port, gps_flag, debugger):
        super(threadEspSocket, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.debugger = debugger
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.ip_address = ip_address
        self.port = port
        self.url = f"{self.ip_address}:{self.port}"
        self.CHECKPOINT = 50
        self.start_flag = True
        self.ready = True
        self.gps_flag = gps_flag
        # self.client.send(self.speed, self.smooth_angle)
        self.subscribe()
        self.Configs()
        self.message = {}
        self.message_type = ""
    
    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadEspSocket", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadEspSocket", "pipe": self.pipeSendConfig},
            }
        )

    # =============================== STOP ================================================
    def stop(self):
        self.socket.close()
        print('Socket Close')
        super(threadEspSocket, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""
        while self.pipeRecvConfig.poll():
            message = self.pipeRecvConfig.recv()
            message = message["value"]
            print(message)
        threading.Timer(1, self.Configs).start()

    def connect(self):
        print(f"Connecting to {self.url}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            run_with_timeout(self.socket.connect, args=(self.ip_address, self.port), timeout=5)
            print(f"Connected to {self.url}")
            self.ready = True
        except:
            print(f"Failed to connect to {self.url}")
            self.connect()
        

    def catch_response(self):
        return self.socket.recv(4096)

    def send(self, req, speed, angle):
        try:
            request = {
                "type": DATATYPES[req],
                "steering": angle,
                "speed": speed,
            }
            json_request = json.dumps(request)
            
            # Send JSON request to server
            self.socket.sendall(json_request.encode('utf-8'))
            print(f"Sent: {json_request}")
            try:
                response = run_with_timeout(self.catch_response, timeout=5)
                response = response.decode('utf-8')
                # if req == 'REQ_GPS':
                self.queuesList[CarControlInfo.Queue.value].put(
                {
                    "Owner": CarControlInfo.Owner.value,
                    "msgID": CarControlInfo.msgID.value,
                    "msgType": CarControlInfo.msgType.value,
                    "msgValue": response,
                })
                self.queuesList[EKFCarControlInfo.Queue.value].put(
                {
                    "Owner": EKFCarControlInfo.Owner.value,
                    "msgID": EKFCarControlInfo.msgID.value,
                    "msgType": EKFCarControlInfo.msgType.value,
                    "msgValue": response,
                })
                self.queuesList[RetrievingInfo.Queue.value].put(
                {
                    "Owner": RetrievingInfo.Owner.value,
                    "msgID": RetrievingInfo.msgID.value,
                    "msgType": RetrievingInfo.msgType.value,
                    "msgValue": response,
                })
                print(f"Received: {response}")
                self.ready = True
            except:
                print('Timed Out. Reconnecting...')
                self.ready = False
                self.reconnect()
                # self.send(speed, angle)
        except:
            print(f"Connection lost. Reconnecting...")
            self.ready = False
            self.reconnect()
            # self.send(speed, angle)
    
    def reconnect(self):
        self.socket.close()
        time.sleep(1)
        self.connect()

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. 
        It captures the image from camera and make the required modifies and then it send the data to process gateway.""" 
        self.connect()  
        while self._running:
            if self.start_flag:
                self.send('GET_ALL_DATA', 0, 0)
                self.start_flag = False
            else:
                ### RUNNING ###
                if not self.queuesList["CarControl"].empty():
                    value = self.queuesList["CarControl"].get()["msgValue"]
                    if self.ready:
                        self.send('GET_ALL_DATA', round(value["Speed"], 2), round(value["Angle"], 2))
                else:
                    if self.ready:
                        self.send('REQ_ALL_DATA', 0, 0)

    # =============================== START ===============================================
    def start(self):
        print("Initialize Model")
        time.sleep(30)
        print('Done')
        super(threadEspSocket, self).start()

        

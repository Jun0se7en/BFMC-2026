import sys

sys.path.append(".")
import logging
import os
import time
from multiprocessing import Event, Queue
from multiprocessing.sharedctypes import Value
from ctypes import c_bool

import torch
import argparse

from src.clearBuffer.processClearBuffer import processClearBuffer

# ===================================== PROCESS IMPORTS ==================================
from src.gateway.processGateway import processGateway
from src.hardware.camera.processCamera import processCamera
from src.imageProcessing.processBoschNet import processBoschNet
from src.server.processServer import processServer
from src.decisionMaking.processDecisionMaking import processDecisionMaking
from src.control.manualControl.processManualControl import processManualControl
from src.hardware.serialhandler.processSerialHandler import processSerialHandler
from src.data.TrafficCommunication.processTrafficCommunication import processTrafficCommunication
from src.EKF.processEKF import processEKF

# ======================================== SETTING UP ====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Input IP')
    parser.add_argument('--xavierip', type=str, help='Xavier IP', default="192.168.7.40")
    parser.add_argument('--xavierport', type=int, help='Xavier Port', default=1234)
    parser.add_argument('--serverip', type=str, help='Localization Server IP', default="192.168.1.111")
    parser.add_argument('--serverport', type=int, help='Localization Server Port', default=9000)
    args = parser.parse_args()

    allProcesses = list()
    queueList = {
        'Control': Queue(),
        "Critical": Queue(),
        "Warning": Queue(),
        "General": Queue(),
        "Config": Queue(),
        # After Processed
        "ObjectDetection": Queue(),
        "Segmentation": Queue(),
        "Points": Queue(),
        "DecisionMaking": Queue(),

        # Camera
        "MainCamera": Queue(),
        "BoschNetCamera": Queue(),

        # Localization
        "Position": Queue(),
        "CarGPSInfo": Queue(),

        "CarStats": Queue(),

        # Distance
        "DistanceData": Queue(),

        # Decision Making
        "EKF": Queue(),
        "AngleLaneKeeping": Queue(),
        "Heading": Queue(),
        "SendHeading": Queue(),
        "SendingSpeed": Queue(),
        "TrafficLog": Queue(),
    }
    
    Speed = Value("i", 0)
    Steer = Value("i", 0)
    Reset = Value(c_bool, False)
    

    logging = logging.getLogger()

    # Camera
    Camera = True
    
    # Image Processing
    BoschNet = True

    # Server
    Server = True
    
    # Clear Buffer (Removed)
    ClearBuffer = True

    # Manual Control
    ManualControl = False

    # DecisionMaking
    DecisionMaking = True
    
    # Serial Handler
    SerialHandler = True
    
    # Traffic Communication
    TrafficCommunication = True

    # EKFSS
    EKF = True

    # =========================== CHECKING NECESSARY PROCESSES ===============================
    # if not Camera:
    #     raise Exception("Camera is not initialized!!!")

    # if (ManualControl or DecisionMaking) and not SerialHandler:
    #     raise Exception("Serial Handler is not initialized!!!")

    # if not ClearBuffer:
    #     raise Exception("Clear Buffer is not initialized!!!")

    # ===================================== SETUP PROCESSES ==================================

    # Initializing gateway
    processGateway = processGateway(queueList, logging)
    allProcesses.append(processGateway)

    # Initializing camera
    if Camera:
        width = 1280.0
        height = 720.0
        fps = 90
        processCamera = processCamera(queueList, logging, width, height, fps, debugging=False)
        allProcesses.append(processCamera)

    if BoschNet:
        processBoschNet = processBoschNet(queueList, logging, Speed, Steer, Reset, debugging=True)
        allProcesses.append(processBoschNet)
    
    # Initializing serial connection NUCLEO - > PI
    if SerialHandler:
        processSerialHandler = processSerialHandler(queueList, logging, Speed, Steer, Reset)
        allProcesses.append(processSerialHandler)

    if Server:
        hostname = args.xavierip # Xavier IP
        port = args.xavierport
        kindofimages = ["Position", "ObjectDetection", "Points", "Segmentation", "CarGPSInfo", "MainCamera"]
        kind = kindofimages[1]
        processServer1 = processServer(
            queueList, logging, hostname, port, kind, debugging=False
        )
        allProcesses.append(processServer1)
        port += 1
        kind = kindofimages[2]
        processServer2 = processServer(
            queueList, logging, hostname, port, kind, debugging=False
        )
        allProcesses.append(processServer2)
        # port += 1
        # kind = kindofimages[4]
        # processServer3 = processServer(
        #     queueList, logging, hostname, port, kind, debugging=False
        # )
        # allProcesses.append(processServer3)

    if ClearBuffer:
        processClearBuffer = processClearBuffer(queueList, logging, debugging=False)
        allProcesses.append(processClearBuffer)

    if DecisionMaking:
        processDecisionMaking = processDecisionMaking(queueList, logging, Speed, Steer, debugging=True)
        allProcesses.append(processDecisionMaking)

    if ManualControl:
        processManualControl = processManualControl(
            queueList, logging, Speed, Steer, Reset, debugging=False
        )
        allProcesses.append(processManualControl)
    
    if TrafficCommunication:
        processTrafficCommunication = processTrafficCommunication(queueList, logging, 10, debugging=False)
        allProcesses.append(processTrafficCommunication)

    if EKF:
        processEKF = processEKF(queueList, Speed, Steer, logging, debugging=False)
        allProcesses.append(processEKF)

    # ===================================== START PROCESSES ==================================
    for process in allProcesses:
        process.daemon = True
        process.start()

    # ===================================== STAYING ALIVE ====================================
    blocker = Event()
    try:
        blocker.wait()
    except KeyboardInterrupt:
        print("\nCatching a Keyboard Interruption exception! Shutdown all processes.\n")
        for proc in allProcesses:
            print("Process stopped", proc)
            proc.stop()
            time.sleep(0.1)
            proc.join()

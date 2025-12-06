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

from multiprocessing import Pipe
from src.utils.messages.allMessages import (
    mainCamera,
    serialCamera,
    Recording,
    Record,
    Config,
    BoschNetCamera,
)
from src.templates.threadwithstop import ThreadWithStop


class threadCamera(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        pipeRecv (multiprocessing.queues.Pipe): A pipe where we can receive configs for camera. We will read from this pipe.
        pipeSend (multiprocessing.queues.Pipe): A pipe where we can write configs for camera. Process Gateway will write on this pipe.
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, width, height, fps, debugger):
        super(threadCamera, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.pipeRecvConfig = pipeRecv
        self.pipeSendConfig = pipeSend
        self.width = width
        self.height = height
        self.fps = fps
        self.debugger = debugger
        self.frame_rate = 5
        self.recording = False
        pipeRecvRecord, pipeSendRecord = Pipe(duplex=False)
        self.pipeRecvRecord = pipeRecvRecord
        self.pipeSendRecord = pipeSendRecord
        self.video_writer = ""
        self.subscribe()
        self._init_camera()
        self.Queue_Sending()
        self.Configs()
        # print('Initialize camera thread!!!')

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Record.Owner.value,
                "msgID": Record.msgID.value,
                "To": {"receiver": "threadCamera", "pipe": self.pipeSendRecord},
            }
        )
        self.queuesList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "To": {"receiver": "threadCamera", "pipe": self.pipeSendConfig},
            }
        )

    def Queue_Sending(self):
        """Callback function for recording flag."""
        self.queuesList[Recording.Queue.value].put(
            {
                "Owner": Recording.Owner.value,
                "msgID": Recording.msgID.value,
                "msgType": Recording.msgType.value,
                "msgValue": self.recording,
            }
        )
        threading.Timer(1, self.Queue_Sending).start()

    # =============================== STOP ================================================
    def stop(self):
        self.camera.release()
        if self.recording:
            self.video_writer.release()
        super(threadCamera, self).stop()

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
        var = True
        while self._running:
            if self.debugger == True:
                self.logger.warning("Getting image!!!")
            start = time.time()
            ret, request = self.camera.read()
            # print('FPS: ', self.camera.get(cv2.CAP_PROP_FPS))
            # print('FPS: ', round(1/(time.time()-start), 1))
            if not ret:
                print("Read failed")
            if var:
                if self.recording == True:
                    cv2_image = cv2.cvtColor(request, cv2.COLOR_RGB2BGR)
                    self.video_writer.write(cv2_image)

                request = cv2.resize(request, (640, 480))
                _, encoded_img = cv2.imencode(".jpg", request)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
                
                # if not self.queuesList[mainCamera.Queue.value].empty():
                #     _ = self.queuesList[mainCamera.Queue.value].get()
                # if not self.queuesList[BoschNetCamera.Queue.value].empty():
                #     _ = self.queuesList[BoschNetCamera.Queue.value].get()
                self.queuesList[mainCamera.Queue.value].put(
                    {
                        "Owner": mainCamera.Owner.value,
                        "msgID": mainCamera.msgID.value,
                        "msgType": mainCamera.msgType.value,
                        "msgValue": image_data_encoded,
                    }
                )
                self.queuesList[BoschNetCamera.Queue.value].put(
                    {
                        "Owner": BoschNetCamera.Owner.value,
                        "msgID": BoschNetCamera.msgID.value,
                        "msgType": BoschNetCamera.msgType.value,
                        "msgValue": image_data_encoded,
                    }
                )
            # print('Camera: ', time.time()-start)
            # print('Camera is running!!!')
            var = not var

    # =============================== START ===============================================
    def start(self):
        print('Waiting for Model to Initialize!~!')
        time.sleep(5)
        print('Done')
        super(threadCamera, self).start()

    # ================================ INIT CAMERA ========================================
    def _init_camera(self):
        # self.camera = cv2.VideoCapture('./test.mp4')
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.camera.isOpened():
            print("Capture failed")
        

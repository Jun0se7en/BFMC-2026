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

import logging
from twisted.internet import task

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class periodicTask(task.LoopingCall):
    def __init__(self, interval, shrd_mem, tcp_factory, queuesList):
        logging.info("Initializing periodicTask")
        super().__init__(self.periodicCheck)
        self.interval = interval
        self.shrd_mem = shrd_mem
        self.queuesList = queuesList
        self.tcp_factory = tcp_factory
        logging.info("periodicTask initialized")

    def start(self):
        """
        Start the periodic task with the specified interval.
        """
        logging.info("Starting periodicTask")
        super().start(self.interval)

    def periodicCheck(self):
        """
        Perform the periodic check and send data to the server.
        """
        if not self.queuesList["Position"].empty():
            msg = self.queuesList["Position"].get()['msgValue']
            self.shrd_mem.insert("devicePos", [msg["x"], msg["y"]])
            self.x = msg["x"]
            self.y = msg["y"]
        
        if not self.queuesList["SendHeading"].empty():
            msg = self.queuesList["SendHeading"].get()['msgValue']
            self.shrd_mem.insert("deviceRot", [msg])
        
        if not self.queuesList["SendingSpeed"].empty():
            msg = self.queuesList["SendingSpeed"].get()['msgValue']
            self.shrd_mem.insert("deviceSpeed", [msg])

        if not self.queuesList["TrafficLog"].empty():
            msg = self.queuesList["TrafficLog"].get()['msgValue']
            if "stop" in msg:
                self.shrd_mem.insert("historyData", [1, self.x, self.y])
            elif "priorityroad" in msg:
                self.shrd_mem.insert("historyData", [2, self.x, self.y])
            elif "parking" in msg:
                self.shrd_mem.insert("historyData", [3, self.x, self.y])
            elif "crosswalk" in msg:
                self.shrd_mem.insert("historyData", [4, self.x, self.y])
            elif "highwayentry" in msg:
                self.shrd_mem.insert("historyData", [5, self.x, self.y])
            elif "highwayend" in msg:
                self.shrd_mem.insert("historyData", [6, self.x, self.y])
            elif "roundabout" in msg:
                self.shrd_mem.insert("historyData", [7, self.x, self.y])
            elif "oneway" in msg:
                self.shrd_mem.insert("historyData", [8, self.x, self.y])
            elif "noentry" in msg:
                self.shrd_mem.insert("historyData", [9, self.x, self.y])
            elif "parking" in msg and "car" in msg:
                self.shrd_mem.insert("historyData", [10, self.x, self.y])
            elif "crosswalk" in msg and "pedestrian" in msg:
                self.shrd_mem.insert("historyData", [11, self.x, self.y])
            elif "pedestrian" in msg:
                self.shrd_mem.insert("historyData", [12, self.x, self.y])
            elif "redlight" in msg or "yellowlight" in msg or "greenlight" in msg:
                self.shrd_mem.insert("historyData", [14, self.x, self.y])
            
            

        # logging.info("Performing periodic check")
        tosend = self.shrd_mem.get()
        for mem in tosend:
            # logging.info(f"Sending data to server: {mem}")
            self.tcp_factory.send_data_to_server(mem)
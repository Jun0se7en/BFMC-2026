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
import threading
from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import (
    BatteryLvl,
    ImuData,
    InstantConsumption,
    EnableButton,
    DistanceData,
)


class threadRead(ThreadWithStop):
    """This thread read the data that NUCLEO send to Raspberry PI.\n

    Args:
        f_serialCon (serial.Serial): Serial connection between the two boards.
        f_logFile (FileHandler): The path to the history file where you can find the logs from the connection.
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
    """

    # ===================================== INIT =========================================
    def __init__(self, f_serialCon, f_logFile, queueList):
        super(threadRead, self).__init__()
        self.serialCon = f_serialCon
        self.logFile = f_logFile
        self.buff = ""
        self.isResponse = False
        self.queuesList = queueList
        self.acumulator = 0
        self.Queue_Sending()

    # ====================================== RUN ==========================================
    def run(self):
        while self._running:
            read_chr = self.serialCon.read()
            try:
                # print("hi: ", read_chr, "\n")
                read_chr = read_chr.decode("ascii")
                # print("hello: ", read_chr, "\n")
                if read_chr == "@":
                    self.isResponse = True
                    if len(self.buff) != 0:
                        self.sendqueue(self.buff)
                    self.buff = ""
                elif read_chr == "\r":
                    self.isResponse = False
                    if len(self.buff) != 0:
                        self.sendqueue(self.buff)
                    self.buff = ""
                if self.isResponse:
                    self.buff += read_chr
            except UnicodeDecodeError:
                print("######### UnicodeDecodeError ###########")
                pass

    # ==================================== SENDING =======================================
    def Queue_Sending(self):
        """Callback function for enable button flag."""
        self.queuesList[EnableButton.Queue.value].put(
            {
                "Owner": EnableButton.Owner.value,
                "msgID": EnableButton.msgID.value,
                "msgType": EnableButton.msgType.value,
                "msgValue": True,
            }
        )
        threading.Timer(1, self.Queue_Sending).start()

    def sendqueue(self, buff):
        """This function select which type of message we receive from NUCLEO and send the data further."""
        # print("Received buffer: ", buff)
        if buff[1] == "1":
            pass
            # print(buff[3:-2])
        elif buff[1] == "2":
            pass
            # print(buff[3:-2])
        elif buff[1] == "3":
            pass
            # print(buff[3:-2])
        elif buff[1] == "4":
            pass
            # print(buff[3:-2])
        elif buff[1] == "5":
            self.queuesList[BatteryLvl.Queue.value].put(
                {
                    "Owner": BatteryLvl.Owner.value,
                    "msgID": BatteryLvl.msgID.value,
                    "msgType": BatteryLvl.msgType.value,
                    "msgValue": int(buff[2:-3]),
                }
            )
        elif buff[1] == "6":
            self.queuesList[InstantConsumption.Queue.value].put(
                {
                    "Owner": InstantConsumption.Owner.value,
                    "msgID": InstantConsumption.msgID.value,
                    "msgType": InstantConsumption.msgType.value,
                    "msgValue": int(buff[2:-3]),
                }
            )
        elif buff[1] == "7":
            buff = buff[3:-2]
            # print(buff)
            if buff[0] == "0":
                return
            splitedBuffer = buff.split(";")
            # print(splitedBuffer)
            if(len(splitedBuffer)>=6):
                data = {
                    "roll": splitedBuffer[0],
                    "pitch": splitedBuffer[1],
                    "yaw": splitedBuffer[2],
                    "accelx": splitedBuffer[3],
                    "accely": splitedBuffer[4],
                    "accelz": splitedBuffer[5],
                }
                self.queuesList[ImuData.Queue.value].put(
                    {
                        "Owner": ImuData.Owner.value,
                        "msgID": ImuData.msgID.value,
                        "msgType": ImuData.msgType.value,
                        "msgValue": data,
                    }
                )
        elif buff[1] == "8":
            buff = buff[3:-2]
            # print("buff: ", buff)
            # if buff[0] == "0":
            #     return
            splitedBuffer = buff.split(";")
            # data = {
            #     "front": splitedBuffer[0],
            #     "right": splitedBuffer[1],
            #     "left": splitedBuffer[2],
            #     "back": splitedBuffer[3],
            # }
            data = {
                "front": splitedBuffer[3],
                "right": splitedBuffer[2],
                "left": splitedBuffer[0],
                "back": splitedBuffer[1],
            }
            # print("data: ", data)
            if not self.queuesList[DistanceData.Queue.value].empty():
                temp = self.queuesList[DistanceData.Queue.value].get()   
            self.queuesList[DistanceData.Queue.value].put({
                "Owner": DistanceData.Owner.value,
                "msgID": DistanceData.msgID.value,
                "msgType": DistanceData.msgType.value,
                "msgValue": data,
            })

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
from enum import Enum


####################################### processCamera #######################################
class mainCamera(Enum):
    Queue = "MainCamera"
    Owner = "threadCamera"
    msgID = 1
    msgType = "base64"


class serialCamera(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 2
    msgType = "base64"


class Recording(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 3
    msgType = "Boolean"


class Signal(Enum):
    Queue = "General"
    Owner = "threadCamera"
    msgID = 4
    msgType = "String"

class BoschNetCamera(Enum):
    Queue = "BoschNetCamera"
    Owner = "threadCamera"
    msgID = 5
    msgType = "base64"

class Segmentation(Enum):
    Queue = "Segmentation"
    Owner = "threadSegmentation"
    msgID = 6
    msgType = "base64"

class Points(Enum):
    Queue = "Points"
    Owner = "threadSegmentation"
    msgID = 7
    msgType = "base64"

class Key(Enum):
    Queue = "Key"
    Owner = "threadServerReceive"
    msgID = 8
    msgType = "int"

class CarControl(Enum):
    Queue = "CarControl"
    Owner = "threadEspSocket"
    msgID = 9
    msgType = "dictionary"

class CarControlInfo(Enum):
    Queue = "CarControlInfo"
    Owner = "threadEspSocket"
    msgID = 10
    msgType = "dictionary"

class EKFCarControlInfo(Enum):
    Queue = "EKFCarControlInfo"
    Owner = "threadEspSocket"
    msgID = 11
    msgType = "dictionary"

class FilterGPS(Enum):
    Queue = "FilterGPS"
    Owner = "threadEKF"
    msgID = 12
    msgType = "dictionary"

class RetrievingInfo(Enum):
    Queue = "RetrievingInfo"
    Owner = "threadRetrieving"
    msgID = 13
    msgType = "dictionary"

class RetrievingCamera(Enum):
    Queue = "RetrievingCamera"
    Owner = "threadRetrieving"
    msgID = 14
    msgType = "base64"

class RetrievingSegment(Enum):
    Queue = "RetrievingSegment"
    Owner = "threadRetrieving"
    msgID = 15
    msgType = "base64"

class ObjectDetection(Enum):
    Queue = "ObjectDetection"
    Owner = "threadBoschNet"
    msgID = 16
    msgType = "dictionary"

class DecisionMaking(Enum):
    Queue = "DecisionMaking"
    Owner = "threadBoschNet"
    msgID = 17
    msgType = "dictionary"

class CarStats(Enum):
    Queue = "CarStats"
    Owner = "threadBoschNet"
    msgID = 18
    msgType = "list"

class CarGPSInfo(Enum):
    Queue = "CarGPSInfo"
    Owner = "threadTrafficCommunication"
    msgID = 19
    msgType = "dictionary"

class AngleLaneKeeping(Enum):
    Queue = "AngleLaneKeeping"
    Owner = "threadLaneKeeping"
    msgID = 20
    msgType = "dictionary"

class EKF(Enum):
    Queue = "EKF"
    Owner = "threadWitMotion"
    msgID = 21
    msgType = "dictionary"

class Heading(Enum):
    Queue = "Heading"
    Owner = "threadWitMotion"
    msgID = 22
    msgType = "float"

class SendHeading(Enum):
    Queue = "SendHeading"
    Owner = "threadWitMotion"
    msgID = 23
    msgType = "float"

class SendingSpeed(Enum):
    Queue = "SendingSpeed"
    Owner = "threadDecisionMaking"
    msgID = 24
    msgType = "float"

class TrafficLog(Enum):
    Queue = "TrafficLog"
    Owner = "threadBoschNet"
    msgID = 25
    msgType = "dictionary"

################################# processCarsAndSemaphores ##################################
class Cars(Enum):
    Queue = "General"
    Owner = "threadCarsAndSemaphores"
    msgID = 1
    msgType = "String"


class Semaphores(Enum):
    Queue = "General"
    Owner = "threadCarsAndSemaphores"
    msgID = 2
    msgType = "String"


################################# From PC ##################################
class EngineRun(Enum):
    Queue = "General"
    Owner = "threadRemoteHandler"
    msgID = 1
    msgType = "dictionary"


# {"action": "startEngine", "value": self.started}


class SpeedMotor(Enum):
    Queue = "General"
    Owner = "threadRemoteHandler"
    msgID = 2
    msgType = "dictionary"


# "action": "speed", "value": val}


class SteerMotor(Enum):
    Queue = "General"
    Owner = "threadRemoteHandler"
    msgID = 3
    msgType = "dictionary"


# {"action": "steer", "value": val}


class Control(Enum):
    Queue = "Control"
    Owner = "threadRemoteHandler"
    msgID = 4
    msgType = "dictionary"


class Brake(Enum):
    Queue = "General"
    Owner = "threadRemoteHandler"
    msgID = 5
    msgType = "dictionary"


# {"action": "steer", "value": 0.0}
# {"action": "speed", "value": 0.0}


class Record(Enum):
    Queue = "General"
    Owner = "threadRemoteHandler"
    msgID = 6
    msgType = "dictionary"


# {"action": "startRecord", "value": self.startedRecord}


class Config(Enum):
    Queue = "General"
    Owner = "threadRemoteHandler"
    msgID = 7
    msgType = "dictionary"


# {"action": key, "value": value}
    
class Speed(Enum):
    Queue = "Speed"
    Owner = "threadImageControl"
    msgID = 8
    msgType = "int64"

class Steer(Enum):
    Queue = "Steer"
    Owner = "threadImageControl"
    msgID = 9
    msgType = "int64"


################################# From Nucleo ##################################
class BatteryLvl(Enum):
    Queue = "General"
    Owner = "threadReadSerial"
    msgID = 1
    msgType = "float"


class ImuData(Enum):
    Queue = "General"
    Owner = "threadReadSerial"
    msgID = 2
    msgType = "String"


class InstantConsumption(Enum):
    Queue = "General"
    Owner = "threadReadSerial"
    msgID = 3
    msgType = "float"

class DistanceData(Enum):
    Queue = "DistanceData"
    Owner = "threadReadSerial"
    msgID = 4
    msgType = "String"
################################# From Locsys ##################################
class Location(Enum):
    Queue = "General"
    Owner = "threadTrafficCommunication"
    msgID = 1
    msgType = "dictionary"


# {"x": value, "y": value}


######################    From processSerialHandler  ###########################
    
class EnableButton(Enum):
    Queue = "General"
    Owner = "threadWrite"
    msgID = 1
    msgType = "Boolean"


class SignalRunning(Enum):
    Queue = "General"
    Owner = "threadWrite"
    msgID = 2
    msgType = "Boolean"

######################    From processPositionFusion  ###########################
class FusedPosition(Enum):
    Queue = "General"
    Owner = "threadUKF"
    msgID = 1
    msgType = "dictionary"

######################    From processTrafficCommunication  ###########################
class Position(Enum):
    Queue = "Position"
    Owner = "threadTrafficCommunication"
    msgID = 1
    msgType = "dictionary"

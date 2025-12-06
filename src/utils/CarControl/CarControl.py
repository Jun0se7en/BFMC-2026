import multiprocessing
import json
from src.utils.messages.allMessages import (
    SignalRunning,
    Config,
    Speed,
    Steer,
    Control,
    DistanceData,
    ImuData, 
    SendingSpeed
)

class CarControl:
    def __init__(self,queueList, Speed, Steer,IMUSendpipe = None,IMURecvpipe = None,VLXSendpipe = None,VLXRecvpipe = None):
        self.IMUSendpipe = IMUSendpipe
        self.VLXSendpipe = VLXSendpipe
        self.IMURecvpipe = IMURecvpipe
        self.VLXRecvpipe = VLXRecvpipe
        self.queueList = queueList
        self.Speed = Speed
        self.Steer = Steer
        self.IMUperiod = 0
        self.VLXperiod = 0
        self.IMUdata = {}
        self.VLXdata = {}
        # self.GateWay_Subcribe()
        
        
    def GateWay_Subcribe(self):
        if self.VLXSendpipe != None:
            self.queuesList[DistanceData.Queue.value].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": DistanceData.Owner.value,
                "msgID": DistanceData.msgID.value,
                "To": {"receiver": "threadWrite", "pipe": self.VLXSendpipe},
            }
        )
        if self.IMUSendpipe != None:
            self.queuesList[ImuData.Queue.value].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": ImuData.Owner.value,
                "msgID": ImuData.msgID.value,
                "To": {"receiver": "threadWrite", "pipe": self.IMUSendpipe},
            }
        )
        
    def setSpeed(self,speed):
        if not self.queueList[SendingSpeed.Queue.value].empty():
            _ = self.queueList[SendingSpeed.Queue.value].get()
        self.queueList[SendingSpeed.Queue.value].put(
            {
                "Owner": SendingSpeed.Owner.value,
                "msgID": SendingSpeed.msgID.value,
                "msgType": SendingSpeed.msgType.value,
                "msgValue": speed,
            }
        )
        self.Speed.value = speed
        
    def getSpeed(self):
        return self.Speed.value
    
    def setAngle(self,angle):
        # print(self.Steer.value)
        self.Steer.value = angle
        
    def getAngle(self):
        return self.Steer.value
    
    def enIMU(self,period):
        self.IMUperiod = period
        self.queueList[Config.Queue.value].put(
            {
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "msgType": Config.msgType.value,
                "msgValue": {
                    'action': 'enable_imu',
                    'value': self.IMUperiod
                },
            }
        )
    
    def disIMU(self):
        self.IMUperiod = 0
        self.queueList[Config.Queue.value].put(
            {
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "msgType": Config.msgType.value,
                "msgValue": {
                    'action': 'enable_imu',
                    'value': self.IMUperiod
                },
            }
        )
    
    def getIMUdata(self):
        if self.IMURecvpipe.poll():
            self.IMUdata = self.IMURecvpipe.recv()
        return self.IMUdata
    
    def enVLX(self,period):
        self.VLXperiod = period
        self.queueList[Config.Queue.value].put(
            {
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "msgType": Config.msgType.value,
                "msgValue": {
                    'action': 'enable_vlx',
                    'value': self.VLXperiod
                },
            }
        )
        
    def disVLX(self,period):
        self.VLXperiod = 0
        self.queueList[Config.Queue.value].put(
            {
                "Owner": Config.Owner.value,
                "msgID": Config.msgID.value,
                "msgType": Config.msgType.value,
                "msgValue": {
                    'action': 'enable_vlx',
                    'value': self.VLXperiod
                },
            }
        )
        
    def getVLXdata(self):
        if not self.queueList[DistanceData.Queue.value].empty():
            self.VLXdata = self.queueList[DistanceData.Queue.value].get()
        return self.VLXdata

    
    def setControl(self,speed,angle,time):
        print('send Control')
        self.queueList[Control.Queue.value].put({
                "Owner": Control.Owner.value,
                "msgID": Control.msgID.value,
                "msgType": Control.msgType.value,
                "msgValue": {
                    'action': '9',
                    'Speed': speed,
                    'Steer': angle,
                    'Time': time
                },
        })
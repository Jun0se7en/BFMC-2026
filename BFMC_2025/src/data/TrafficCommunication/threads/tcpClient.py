import json
import time
import logging
from src.utils.messages.allMessages import Location
from src.utils.messages.messageHandlerSender import messageHandlerSender
from twisted.internet import protocol
from src.utils.messages.allMessages import (
    Position,
    CarGPSInfo,
)
from scipy.spatial.transform import Rotation as R
import pandas as pd
from scipy.optimize import minimize
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



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def procrustes_transform(P, Q):
    # P, Q shape = (N,2)
    # Center data
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    # SVD
    U, _, Vt = np.linalg.svd(P_centered.T @ Q_centered)
    R = U @ Vt

    # Scale
    scale = np.trace(Q_centered.T @ P_centered @ R) / np.trace(P_centered.T @ P_centered)

    # R_fix = np.eye(2)

    # Translation
    t = Q.mean(axis=0) - scale * (R @ P.mean(axis=0))

    return scale, R, t

# Ánh xạ điểm mới p:
def map_point(p, scale, R, t):
    return scale * (p @ R.T) + t

# The server itself. Creates a new Protocol for each new connection and has the info for all of them.
class tcpClient(protocol.ClientFactory):
    def __init__(self, connectionBrokenCllbck, locsysID, locsysFrequency, queue):
        logging.info("Initializing tcpClient")
        self.connectiondata = None
        self.connection = None
        self.retry_delay = 1
        self.connectionBrokenCllbck = connectionBrokenCllbck
        self.locsysID = locsysID
        self.locsysFrequency = locsysFrequency
        self.queue = queue
        self.sendLocation = messageHandlerSender(self.queue, Location)
        logging.info("tcpClient initialized")

    def clientConnectionLost(self, connector, reason):
        logging.warning(f"Connection lost with server {self.connectiondata}")
        try:
            self.connectiondata = None
            self.connection = None
            self.connectionBrokenCllbck()
        except Exception as e:
            logging.error(f"Error in clientConnectionLost: {e}")

    def clientConnectionFailed(self, connector, reason):
        logging.warning(f"Connection failed. Retrying in {self.retry_delay} seconds... Possible server down or incorrect IP:port match")
        time.sleep(self.retry_delay)
        connector.connect()

    def buildProtocol(self, addr):
        logging.info("Building protocol")
        conn = SingleConnection(self.queue)
        conn.factory = self
        return conn

    def send_data_to_server(self, message):
        # logging.info("Sending data to server")
        if self.connection is not None:
            self.connection.send_data(message)


# One class is generated for each new connection
class SingleConnection(protocol.Protocol):
    def __init__(self, queue):
        super(SingleConnection, self).__init__()
        self.queue = queue

        ################## MAPPING ANCHOR ################
        real_anchor = []
        
        # Load the CSV file
        for i in range(1, 9):
            df = pd.read_csv(f'./points/point{i}.csv')
            real_anchor.append((np.mean(df['x']), np.mean(df['y']), np.mean(df['z'])))

        real_anchor = np.array(real_anchor, dtype=float)

        # print(real_anchor)

        ### Rotation and Scaling ###
        rotation_degrees = 153
        rotation_radians = np.radians(rotation_degrees)
        rotation_axis = np.array([0, 0, 1])

        rotation_vector = rotation_radians * rotation_axis
        self.rotation = R.from_rotvec(rotation_vector)

        rotation_anchor = self.rotation.apply(real_anchor)

        map_anchor = [(21, 380), (212, 244), (244, 185), (186, 154), (154, 212), (378, 20), (154, 365), (154, 59)]
        map_anchor = np.array(map_anchor, dtype=float)  # (N,2)
        rotation_anchor = rotation_anchor[:, :2]  # Chỉ lấy 2D (bỏ z)
        rotation_anchor = np.array(rotation_anchor, dtype=float)  # (N,2)
        
        self.scale, self.R, self.t = procrustes_transform(rotation_anchor, map_anchor)

        self.coords = []
        
    def connectionMade(self):
        logging.info("Connection made")
        peer = self.transport.getPeer()
        self.factory.connectiondata = peer.host + ":" + str(peer.port)
        self.factory.connection = self
        self.subscribeToLocaitonData(self.factory.locsysID, self.factory.locsysFrequency)
        logging.info(f"Connection with server established: {self.factory.connectiondata}")

    def dataReceived(self, data):
        # logging.info("Data received")
        dat = data.decode()
        tmp_data = dat.replace("}{","}}{{")
        if tmp_data != dat:
            tmp_dat = tmp_data.split("}{")
            dat = tmp_dat[-1]
        da = json.loads(dat)
        da["x"] = da["x"] / 1000
        da["y"] = da["y"] / 1000
        da["z"] = da["z"] / 1000
        # print(da)
        # rotation_points = self.rotation.apply([(da["x"], da["y"], da["z"])])
        # rotation_points = rotation_points[:, :2]
        # transformed_points = map_point(rotation_points, self.scale, self.R, self.t)
        # x, y = transformed_points[:, 0], 400 - transformed_points[:, 1]
        # da["x"] = x[0]
        # da["y"] = y[0]
        # if len(self.coords) < 10:
        #     self.coords.append((da["x"], da["y"]))
        # else:
        #     da["x"] = np.mean([coord[0] for coord in self.coords])
        #     da["y"] = np.mean([coord[1] for coord in self.coords])
        #     self.coords.clear()
        #     if not self.queue[CarGPSInfo.Queue.value].empty():
        #         _ = self.queue[CarGPSInfo.Queue.value].get()

        #     self.queue[CarGPSInfo.Queue.value].put(
        #         {
        #             "Owner": CarGPSInfo.Owner.value,
        #             "msgID": CarGPSInfo.msgID.value,
        #             "msgType": CarGPSInfo.msgType.value,
        #             "msgValue": da,
        #         }
        #     )
        
        if not self.queue[Position.Queue.value].empty():
            _ = self.queue[Position.Queue.value].get()

        self.queue[Position.Queue.value].put(
            {
                "Owner": Position.Owner.value,
                "msgID": Position.msgID.value,
                "msgType": Position.msgType.value,
                "msgValue": da,
            }
        )

        if not self.queue[CarGPSInfo.Queue.value].empty():
            _ = self.queue[CarGPSInfo.Queue.value].get()

        self.queue[CarGPSInfo.Queue.value].put(
            {
                "Owner": CarGPSInfo.Owner.value,
                "msgID": CarGPSInfo.msgID.value,
                "msgType": CarGPSInfo.msgType.value,
                "msgValue": da,
            }
        )

        # print(da)

        if da["type"] == "location":
            da["id"] = self.factory.locsysID
            self.factory.sendLocation.send(da)
        else:
            logging.info(f"Got message from traffic communication server: {self.factory.connectiondata}")

    def send_data(self, message):
        # logging.info("Sending data")
        msg = json.dumps(message)
        self.transport.write(msg.encode())
    
    def subscribeToLocaitonData(self, id, frequency):
        logging.info("Subscribing to location data")
        # Sends the id you wish to subscribe to and the frequency you want to receive data. Frequency must be between 0.1 and 5. 
        msg = {
            "reqORinfo": "info",
            "type": "locIDsub",
            "locID": id,
            "freq": frequency,
        }
        self.send_data(msg)
    
    def unSubscribeToLocaitonData(self, id, frequency):
        logging.info("Unsubscribing from location data")
        # Unsubscribes from location data. 
        msg = {
            "reqORinfo": "info",
            "type": "locIDubsub",
        }
        self.send_data(msg)
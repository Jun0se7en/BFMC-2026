import cv2
import threading
import base64
import time
import json
import collections
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
from src.imageProcessing.InferenceBoschNet import InferenceBoschNet
from src.imageProcessing.CarInstruction import CarInstruction
from src.imageProcessing.DirectionPicking import DirectionPicking

from lib.LaneKeeping4 import LaneKeeping
from src.utils.CarControl.CarControl import CarControl

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
    def __init__(self, pipeRecv, pipeSend, queuesList, logger, Speed, Steer, debugger):
        super(threadBoschNet, self).__init__()
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
    
        self.lanekeeping = LaneKeeping()
        self.Beta = 0.5
        self.Speed, self.Steer = Speed, Steer
        self.speed, self.angle = 0, 0
        self.control = CarControl(self.queuesList, self.Speed, self.Steer)

        self.width = 640.0
        self.height = 480.0
        self.fps = 30
        self._init_camera()
        
        self.inference_boschnet = InferenceBoschNet(model_file='./models/model_4.engine', input_size=(640, 384), debugger=debugger)
        
        # DecisionMaking
        event_cfg = 'signs_area.json'
        nodes_cfg = 'map_nodes.json'
        
        self.direction_picking = DirectionPicking(config_file=nodes_cfg)
        self.car_instruction = CarInstruction()
        self.sign_config = self.load_config_file(event_cfg)
        self.sign_actions = self.car_instruction.sign_actions
        self.intersection_actions = self.car_instruction.intersection_actions
        
        self.recorded_time = 0
        self.flag_lanekeeping = True
        self.flag_is_intersection = False
        self.queue_keeping = collections.deque([False] * 7, maxlen=7)
        
        self.flag_adjust_speed = False
        self.plus_speed = 0
        self.speed_signs = ['Pedestrian', 'HighwayEntry', 'HighwayEnd', 'Redlight', 'Yellowlight', 'CrossWalk']
        self.is_danger = False
        
        # possible direction at intersection: ['go_forwward', 'go_left', 'go_right', 'stop']
        self.direction_lst = ['go_forward', 'go_left', 'go_right', 'stop']
        self.direction_idx = 0
        self.last_angle = 0
        self.cooldown = 0
        
        self.start_node = 1
        self.destination_node = 13
        
    
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
        
    # ================================ INSTRUCTION CONFIG ================================================
    @staticmethod
    def load_config_file(config_file):
        with open(config_file, "r") as jsonfile:
            data = json.load(jsonfile)
        return data
    
    def get_dict(self, obj_msg):
        for key, value in self.sign_config.items():
            if value.get("obj_msg") == obj_msg:
                # return name of function and area threshold
                return key, value.get("area")
        return None
                        
    def obj_filter(self, obj_msg):
        obj_lst = obj_msg.get("Class", [])
        area_lst = obj_msg.get("Area", [])
        
        if not area_lst or not obj_lst or len(obj_lst) != len(area_lst):
            return None
        
        max_index = area_lst.index(max(area_lst))
        curr_area = area_lst[max_index]
        obj_class = obj_lst[max_index]
        
        return (obj_class, curr_area)
    
    # This function include 
    def exec_instruction(self, instruction_set):
        # If it is strictly instruction
        if not instruction_set[1]:
            # print('Len of instruction: ', len(instruction_set[0]))
            for i in instruction_set[0]: # Loop through each instruction
                self.flag_lanekeeping = False 
                speed, angle, time_sec = i
                self.speed, self.angle = speed, angle
                self.control.setControl(self.speed, self.angle, 1)
                time.sleep(time_sec)
            
            self.flag_lanekeeping = True         
        # Adjust only plus_speed
        else:
            self.flag_lanekeeping = True
            self.plus_speed = int(instruction_set[0][0][0]) # Get only speed value
            
    def get_sign_attr(self, filtered_obj_msg):
            obj_class, curr_area = filtered_obj_msg
            # Get threshold from json file 
            result = self.get_dict(obj_class)

            sign_msg, thresh_area = result

            # Check the confidence time of the sign, and if it exceed area threshold
            if curr_area > thresh_area:
                if self.recorded_time == 0:
                    self.recorded_time = time.time()
                    return None
                if time.time() - self.recorded_time > 0.15:
                    instruction_set = self.sign_actions.get(sign_msg, lambda: None)()
                    self.recorded_time = 0
                    return instruction_set
            else:
                self.recorded_time = 0

            return None

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""
        while self._running:
            start = time.time()
            ret, request = self.camera.read()
            if not ret:
                print("Read failed")
                break
            obj_msg = {}
            lane_img, output_obj =  self.inference_boschnet.inference(request)
            img_obj, classes, areas = output_obj
            
            _, encoded_img = cv2.imencode(".jpg", img_obj)
            image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
            
            obj_msg = {"Image": image_data_encoded, "Class": classes, "Area": areas}
            obj_msg2 = {"Class": classes, "Area": areas}
            # if len(classes) != 0:
                # print(classes, areas)
            rc_obj_msg = self.obj_filter(obj_msg2)

            if rc_obj_msg is not None:
                instruction_set = self.get_sign_attr(rc_obj_msg)
                if instruction_set is not None: # If the given sign has instruction
                    obj_class, curr_area = rc_obj_msg
                    if obj_class in self.speed_signs:
                        # print('Speed sign: ', rc_obj_msg)
                        self.flag_adjust_speed = True
                        
                    print("Object message: ", rc_obj_msg)
                    # Give instruction to the car
                    self.exec_instruction(instruction_set)
                    
            else:
                # If passed red light or pedestrian, reset everything
                if self.is_danger:
                    self.plus_speed = 0
                    self.speed = 40
                    self.flag_adjust_speed = False
                    self.is_danger = False
                # print("No object detected")
                
            #### Intersection Condition Check ####      
            if self.flag_is_intersection:
                if time.time() - self.cooldown < 7:
                    print('Cooldown', time.time() - self.cooldown)
                    self.flag_lanekeeping = True
                    self.flag_is_intersection = False
                    self.last_angle = self.angle
                    continue
                
                self.direction_idx = 2 #<-------- mod this part --------
                print('Intersection', self.direction_lst[self.direction_idx])
                direction = self.direction_lst[self.direction_idx]
                instruction_set = self.intersection_actions.get(direction, lambda: None)()
                self.exec_instruction(instruction_set)
                self.flag_lanekeeping = True
                self.flag_is_intersection = False
                self.queue_keeping = collections.deque([False] * 7, maxlen=7)
                self.cooldown = time.time()
                print("Lane keeping mode")
                print(self.last_angle)
            else:
                self.last_angle = self.angle
                
            #### Lane Keeping ####
            if self.flag_lanekeeping:
                self.speed, self.angle, img, robustness = self.lanekeeping.AngCal(lane_img)
                self.queue_keeping.append(robustness)
                if self.flag_adjust_speed:
                    self.speed = self.plus_speed

                self.angle -= self.Beta * (self.angle - self.angle)
                self.angle = int(self.angle + 0.5)
                self.Queue_Sending()

                if self.speed == 0:
                    self.is_danger = True            

            if all(self.queue_keeping):
                self.flag_lanekeeping = False
                self.flag_is_intersection = True
                print("Intersection mode")

            #### Send to Queue ####
            _, encoded_img = cv2.imencode(".jpg", lane_img)
            image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
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
            if not self.flag_lanekeeping:
                _, encoded_img = cv2.imencode(".jpg", lane_img)
                image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
            else:
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
                # print("FPS: ", 1/(time.time()-start))
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
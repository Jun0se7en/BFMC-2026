# import json
# import time
# from src.utils.CarControl.CarControl import CarControl
# from src.decisionMaking.TrafficSignInstruction import TrafficSignInstruction

# class EventHandler:
#     def __init__(self, config_file, queuesList, Speed, Steer):
#         self.config = self.load_config_file(config_file)
#         self.queuesList = queuesList
#         self.Speed, self.Steer = Speed, Steer  # Set these first
#         self.control = CarControl(self.queuesList, self.Speed, self.Steer)
#         self.speed, self.angle = 0, 0
#         self.traffic_sign_instruction = TrafficSignInstruction()
#         self.sign_actions = self.traffic_sign_instruction.sign_actions
        
#         self.recorded_time = 0
#         self.trusted_sign = False
    
#     def queue_sending(self):
#         self.control.setSpeed(self.speed)
#         self.control.setAngle(self.angle)
    
#     @staticmethod
#     def load_config_file(config_file):
#         with open(config_file, "r") as jsonfile:
#             data = json.load(jsonfile)
#         return data
    
#     def get_dict(self, obj_msg):
#         for key, value in self.config.items():
#             if value.get("obj_msg") == obj_msg:
#                 # return name of function and area threshold
#                 return key, value.get("area")
#         return None
    
#     def get_sign_attr(self, filtered_obj_msg):
#         obj_class, curr_area = filtered_obj_msg 
#         result = self.get_dict(obj_class)

#         if result is None:
#             self.recorded_time = 0
#             return None

#         sign_msg, thresh_area = result

#         if curr_area > thresh_area:
#             if self.recorded_time == 0:
#                 self.recorded_time = time.time()
#                 return None
#             if time.time() - self.recorded_time > 0.7:
#                 instruction_set = self.sign_actions.get(sign_msg, lambda: None)()
#                 self.recorded_time = 0
#                 return instruction_set
#         else:
#             self.recorded_time = 0

#         return None
    
#     def exec_instruction(self, instruction_set):
#         for speed_angle, time_sec in instruction_set:
#             speed, angle = speed_angle
#             if all(value is not None for value in (speed, angle, time_sec)):   
#                 self.speed = speed
#                 self.angle = angle
#                 self.control.setControl(self.speed, self.angle, 1)
#                 time.sleep(time_sec)
                
#     def obj_filter(self, obj_msg):
#         obj_lst = obj_msg.get("Class", [])
#         area_lst = obj_msg.get("Area", [])
        
#         if not area_lst or not obj_lst or len(obj_lst) != len(area_lst):
#             return None
        
#         max_index = area_lst.index(max(area_lst))
#         curr_area = area_lst[max_index]
#         obj_class = obj_lst[max_index]
        
#         return (obj_class, curr_area)
        
#     # Call this, output is a set of instruction affect immediately to the engine
#     def recieve_msg(self, obj_msg):
#         filtered_obj_msg = self.obj_filter(obj_msg)     
#         if filtered_obj_msg is not None:  
#             instruction_set = self.get_sign_attr(filtered_obj_msg)
#             if instruction_set is not None:
#                 self.exec_instruction(instruction_set)

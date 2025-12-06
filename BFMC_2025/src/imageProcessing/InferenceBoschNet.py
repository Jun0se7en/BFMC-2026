import torch
import cv2
import numpy as np
from src.utils.utils import non_max_suppression, scale_coords, plot_one_box
from src.imageProcessing.threads.infer_trt import TRT

class InferenceBoschNet:
    """
    InferenceBoschNet performs inference on input images using a TensorRT engine.
    
    It outputs both a processed lane-keeping image and detection results,
    including bounding boxes and class names.
    
    Attributes:
        names (dict): Mapping of class indices to class names.
        colors (list): List of BGR colors for plotting bounding boxes.
        input_size (tuple): Expected input dimensions (width, height) for the model.
        model (TRT): The TensorRT inference engine.
        debugger (bool): If True, draws bounding boxes on the image.
    """
    
    def __init__(self, model_file='./models/model_14.engine', input_size=(640, 384), debugger=False):
        # Class names and their corresponding colors (BGR)
        self.names = {
            0: 'car',
            1: 'crosswalk',
            2: 'greenlight',
            3: 'highwayend',
            4: 'highwayentry',
            5: 'noentry',
            6: 'oneway',
            7: 'parking',
            8: 'pedestrian',
            9: 'priorityroad',
            10: 'redlight',
            11: 'roundabout',
            12: 'stop',
            13: 'yellowlight',
        }
        self.colors = [
            (255, 0, 0),      # Car - Red
            (0, 255, 0),      # CrossWalk - Green
            (0, 0, 255),      # Greenlight - Blue
            (255, 255, 0),    # HighwayEnd - Yellow
            (255, 165, 0),    # HighwayEntry - Orange
            (128, 0, 128),    # NoEntry - Purple
            (0, 255, 255),    # OneWay - Cyan
            (255, 192, 203),  # Parking - Pink
            (139, 69, 19),    # Pedestrian - Brown
            (128, 128, 0),    # PriorityRoad - Olive
            (255, 69, 0),     # Redlight - Red-Orange
            (0, 128, 128),    # Roundabout - Teal
            (169, 169, 169),  # Stop - DarkGray
            (255, 215, 0)     # Yellowlight - Gold
        ]
        
        # Model input dimensions
        self.input_size = input_size  # (width, height)
        self.W_, self.H_ = input_size
        
        # Initialize the TensorRT model
        self.model = TRT(model_file)
        self.debugger = debugger
        
    def inference(self, request):
        # Resize the input image.
        img = cv2.resize(request, (self.W_, self.H_))
        img_obj = img.copy()
        img_rs = np.zeros((self.H_, self.W_, 3))

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img= torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)  # add a batch dimension
        img=img.cuda().float() / 255.0
        img = img.cuda()

        img_out = self.model(img)

        inf_out = img_out[1]
        # print("Before NMS")
        det_pred = non_max_suppression(inf_out, conf_thres=0.7, iou_thres=0.7, classes=None, agnostic=True)
        # print("After NMS")
        det=det_pred[0]

        x0=img_out[0]

        _,da_predict=torch.max(x0, 1)

        DA = da_predict.byte().cpu().data.numpy()[0]*255
        
        img_rs[DA>100]=[255, 255, 255]
        if img_rs.dtype == np.float64:  
            img_rs = cv2.convertScaleAbs(img_rs) 
        img_rs = cv2.resize(img_rs, (320, 240))
        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img_rs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_image=np.zeros((240,320,3),np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(black_image, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        black_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        black_image = cv2.morphologyEx(black_image, cv2.MORPH_CLOSE, kernel)
        black_image = cv2.erode(black_image, kernel, iterations = 3)
        black_image = cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR)
        img_rs = black_image
        # img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        # print(det.shape)
        classes = []
        areas = []
        if len(det) and self.debugger:
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_obj.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{self.names[int(cls)]} {conf:.2f}'
                # print(label_det_pred)
                classes.append(self.names[int(cls)])
                x_min, y_min, x_max, y_max = xyxy
                area = (x_max - x_min) * (y_max - y_min)
                area = area.item()
                areas.append(area)
                plot_one_box(xyxy, img_obj , label=label_det_pred, color=self.colors[int(cls)], line_thickness=2)
        
        else:
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_obj.shape).round()
            for *xyxy,conf,cls in reversed(det):
                classes.append(self.names[int(cls)])
                x_min, y_min, x_max, y_max = xyxy
                area = (x_max - x_min) * (y_max - y_min)
                area = area.item()
                areas.append(area)
        
        # Return the lane image and a tuple of detection results.
        # img_rs: Segment with lane
        # img_obj: raw img with object bboc
        # classes: msg, classes of objects
        # areas: msg, object's area
        return [img_rs, (img_obj, classes, areas)]

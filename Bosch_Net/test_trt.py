import numpy as np
import os
import torch
import tensorrt as trt
from collections import OrderedDict, namedtuple
import torch.nn as nn
import shutil
import cv2
from numpy import random
import time

from utils import non_max_suppression, scale_coords, plot_one_box
def Run(model,img):
    names = {
        0: 'Car',
        1: 'CrossWalk',
        2: 'Greenlight',
        3: 'HighwayEnd',
        4: 'HighwayEntry',
        5: 'NoEntry',
        6: 'OneWay',
        7: 'Parking',
        8: 'Pedestrian',
        9: 'PriorityRoad',
        10: 'Redlight',
        11: 'Roundabout',
        12: 'Stop',
        13: 'Yellowlight',
    }
    colors = [
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
    img = cv2.resize(img, (640, 384))
    img_rs=img.copy()

    
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img=torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img=img.cuda().float() / 255.0
    img = img.cuda()

    # start = time.time()
    img_out = model(img)
    # print("FPS:",1/(time.time()-start))
    # return img_out
    inf_out = img_out[1]
    # print(inf_out.shape)
    det_pred = non_max_suppression(inf_out, conf_thres=0.5, iou_thres=0.6, classes=None, agnostic=True)
    # print(det_pred.shape)
    det=det_pred[0]
    x0=img_out[0]
    
    # print(img_out[0].shape, img_out[1].shape, img_out[2].shape, img_out[3].shape)

    _,da_predict=torch.max(x0, 1)

    DA = da_predict.byte().cpu().data.numpy()[0]*255
    
    img_rs[DA>100]=[117, 240, 107]
    # print(det.shape)
    if len(det):
        det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_rs.shape).round()
        for *xyxy,conf,cls in reversed(det):
            # print(*xyxy)
            x_min, y_min, x_max, y_max = xyxy
            area = (x_max - x_min) * (y_max - y_min)
            # print(area.item())
            label_det_pred = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img_rs , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
    
    return img_rs

import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
class TRT(nn.Module):
    def __init__(self,weight='model_best.engine'):
        super().__init__()
        device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(weight, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.context = model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        for i in range(model.num_bindings):
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            # print(model.get_tensor_mode(name))
            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_tensor_shape(name))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
    def forward(self, im):
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        return [self.bindings[x].data for x in sorted(self.output_names)]

if __name__ == '__main__':
    trt=TRT('./checkpoints/model_4.engine')

    image_list=os.listdir('./inference/images')
    shutil.rmtree('./inference/results')
    os.mkdir('./inference/results')
    for i, imgName in enumerate(image_list):
        img = cv2.imread(os.path.join('./inference/images',imgName))
        # start = time.time()
        img=Run(trt,img)
        # print("FPS:", 1/(time.time()-start))
        # print(len(img))
        cv2.imwrite(os.path.join('./inference/results',imgName),img)

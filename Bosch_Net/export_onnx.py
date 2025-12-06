import torch
import numpy as np
import argparse
# from tqdm.autonotebook import tqdm
import os
import torch
import onnx
import onnxruntime as ort

import pandas as pd
from pathlib import Path

import torch
import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from models.bosch_model import *



if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net4(p = 1, q = 1, n = 1, planes = 4, classes = 14, name = "n")
    checkpoint = torch.load('./checkpoints/model_14.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Load pth done!")
    onnx_path = f'./checkpoints/model_14.onnx'
    inputs = torch.randn((1, 3, 384, 640))
    with torch.no_grad():
        output = model(inputs)
        output1, output2 = model(inputs)
    # print(Da_fmap.shape)
    # for i,m in enumerate(model_out):
    #     print(m.shape,i)
    # out = model(inputs)
    print(f"Converting to {onnx_path}")
    torch.onnx.export(model, inputs, onnx_path,
                      verbose=False, opset_version=12, input_names=['images'],
                      output_names=['da', 'det'])
    print('convert', onnx_path, 'to onnx finish!!!')
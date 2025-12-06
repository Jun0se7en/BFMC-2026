import tensorrt as trt
from pathlib import Path
import torch
import numpy as np
import argparse
# from tqdm.autonotebook import tqdm
import os
import onnx
import pandas as pd
if __name__ == '__main__':
    im = torch.zeros(1, 3, *[1, 3, 384, 640]).to('cuda')
    file = Path('./checkpoints/model_14.onnx')
    onnx = file.with_suffix('.onnx')
    f = file.with_suffix('.engine')  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)

    # logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * 1 << 30
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(
            f' input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(
            f'output "{out.name}" with shape{out.shape} {out.dtype}')


    print(
        f'building FP {16} engine as {f}')
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())
import tensorrt as trt
import numpy as np
import torch
from collections import OrderedDict, namedtuple
import torch.nn as nn

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
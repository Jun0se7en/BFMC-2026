import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from models.common import *
from models.utils.autoanchor import *
from models.utils.utils import *
# from common import *
# from utils.autoanchor import *
# from utils.utils import *
import logging
import os
from thop import profile

class ESP_Encoder(nn.Module):
    def __init__(self, p = 1, q = 1, planes = 4):
        super(ESP_Encoder, self).__init__()

        self.level1 = CBR(3, planes, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = CBR(planes + 3, planes + 3, 3)
        self.level2_0 = DownSamplerB(planes + 3, planes * 2)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(planes * 2, planes * 2))
        self.b2 = CBR(planes * 4 + 3, planes * 4 + 3, 3)

        self.level3_0 = DownSamplerB(planes * 4 + 3, planes * 8)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(planes * 8, planes * 8))
        # self.mixstyle = MixStyle2(p=0.5, alpha=0.1)
        self.b3 = CBR(planes * 16, planes * 4, 3)
    
    def forward(self, input):
        
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        cat_=torch.cat([output2_0, output2], 1)

        output = self.b3(cat_)

        return output, inp1, inp2

class Net1(nn.Module):
    def __init__(self, p = 1, q = 1, n = 3, planes = 4, classes = 10, name = 'n'):
        super(Net1, self).__init__()
        self.name = name
        
        self.dualbranch_encoder = ESP_Encoder(p = p, q = q, planes = planes)
        
        ### CAAM ###
        self.caam = CAAM(feat_in=planes * 4, num_classes= planes * 4, bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        
        ### Segmentation Task ###
        self.up_1_da = UpConvBlock(planes * 4, planes * 2) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(planes * 2, planes) # out: Hx2, Wx2
        self.out_da = UpConvBlock(planes, 2, last=True)  

        ### Object Detection Task ###
        self.down1 = InputProjectionA(1)
        self.down2 = InputProjectionA(2)
        
        self.cbr1 = CBR(planes * 4, planes * 4, 3, 2)
        self.cbr2 = CBR(planes * 4, planes * 4, 3, 2)

        self.spp = SPP(planes * 4,planes * 4)
        self.fpn = FPN(in_channels_list=[planes * 4,planes * 4,planes * 4], out_channels=planes * 2)
        self.pan = PAN(input_channel=planes * 2, output_channel=planes * 2)

        self.detect = Detect(classes, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [planes * 2, planes * 2, planes * 2])
        if isinstance(self.detect, Detect):
            s = 128  # 2x min stride
            self.eval()
            self.detect.train()
            with torch.no_grad():
                _,model_out = self.forward(torch.zeros(1, 3, s, s))
                detects= model_out
                self.detect.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            self.detect.anchors /= self.detect.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(self.detect)
            self.stride = self.detect.stride
            self._initialize_biases()
            self.train()
        
        initialize_weights(self)

    def forward(self, x):
        start = time.time()
        output_encoder, inp1, inp2 = self.dualbranch_encoder(x)
        # print('Encoder Delay: ', time.time()-start)

        spp = self.caam(output_encoder)

        ### Segmentation Task ###
        out_da = self.up_1_da(spp, inp2)
        out_da = self.up_2_da(out_da, inp1)
        out_da = self.out_da(out_da)

        ### Object Detection Task ###
        out1 = self.cbr1(output_encoder)
        out2 = self.cbr2(out1)
        out2 = self.spp(out2)

        out1, out2, out3 = self.fpn([out2, out1, output_encoder])

        out1, out2, out3 = self.pan([out3, out2, out1])
        out_de = self.detect([out1,out2,out3])

        return out_da, out_de
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

class Net2(nn.Module):
    def __init__(self, p = 1, q = 1, n = 3, planes = 4, classes = 10, name = 'n'):
        super(Net2, self).__init__()
        self.name = name
        
        self.dualbranch_encoder = ESP_Encoder(p = p, q = q, planes = planes)
        
        ### CAAM ###
        self.caam = CAAM(feat_in=planes * 4, num_classes= planes * 4, bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        
        ### Segmentation Task ###
        self.up_1_da = UpConvBlock(planes * 4, planes * 2) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(planes * 2, planes) # out: Hx2, Wx2
        self.out_da = UpConvBlock(planes, 2, last=True)  

        ### Object Detection Task ###
        self.cbr1 = Convv5(planes * 4, planes * 8, 3, 2)
        self.cbr1_c3 = nn.ModuleList()
        for i in range(0, n):
            self.cbr1_c3.append(C3(planes * 8, planes * 8))
        
        self.cbr2 = Convv5(planes * 8, planes * 16, 3, 2)
        self.cbr2_c3 = nn.ModuleList()
        for i in range(0, n):
            self.cbr2_c3.append(C3(planes * 16, planes * 16))
        
        self.sppf = SPPF(planes * 16, planes * 16, 5)
        
        self.up1_cbr = Convv5(planes * 16, planes * 8, 1, 1)
        self.up1 = nn.Upsample(scale_factor = 2, mode = "nearest")
        self.up1_c3_0 = C3(planes * 16, planes * 8, shortcut=False)
        self.up1_c3 = nn.ModuleList()
        for i in range(1, n):
            self.up1_c3.append(C3(planes * 8, planes * 8, shortcut=False))
        
        self.up2_cbr = Convv5(planes * 8, planes * 4, 1, 1)
        self.up2 = nn.Upsample(scale_factor = 2, mode="nearest")
        self.up2_c3_0 = C3(planes * 8, planes * 4, shortcut=False)
        self.up2_c3 = nn.ModuleList()
        for i in range(1, n):
            self.up2_c3.append(C3(planes * 4, planes * 4, shortcut=False))
        
        self.down1_cbr = Convv5(planes * 4, planes * 4, 3, 2)
        self.down1_c3 = nn.ModuleList()
        for i in range(0, n):
            self.down1_c3.append(C3(planes * 8, planes * 8, shortcut=False))
        
        self.down2_cbr = Convv5(planes * 8, planes * 8, 3, 2)
        self.down2_c3_0 = C3(planes * 16, planes * 32, shortcut=False)
        self.down2_c3 = nn.ModuleList()
        for i in range(1, n):
            self.down2_c3.append(C3(planes * 32, planes * 32, shortcut=False))    
        

        self.detect = Detect(classes, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [planes * 4, planes * 8, planes * 32])
        if isinstance(self.detect, Detect):
            s = 128  # 2x min stride
            self.eval()
            self.detect.train()
            with torch.no_grad():
                _,model_out = self.forward(torch.zeros(1, 3, s, s))
                detects= model_out
                self.detect.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            self.detect.anchors /= self.detect.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(self.detect)
            self.stride = self.detect.stride
            self._initialize_biases()
            self.train()
        
        initialize_weights(self)

    def forward(self, x):
        output_encoder, inp1, inp2 = self.dualbranch_encoder(x)
        # print('Encoder Delay: ', time.time()-start)

        spp = self.caam(output_encoder)

        ### Segmentation Task ###
        out_da = self.up_1_da(spp, inp2)
        out_da = self.up_2_da(out_da, inp1)
        out_da = self.out_da(out_da)

        ### Object Detection Task ###
        cbr1 = self.cbr1(output_encoder)
        for i, layer in enumerate(self.cbr1_c3):
            if i==0:
                cbr1_c3 = layer(cbr1)
            else:
                cbr1_c3 = layer(cbr1_c3)
        
        cbr2 = self.cbr2(cbr1_c3)
        for i, layer in enumerate(self.cbr2_c3):
            if i==0:
                cbr2_c3 = layer(cbr2)
            else:
                cbr2_c3 = layer(cbr2_c3)
        
        sppf = self.sppf(cbr2_c3)
        
        up1_cbr = self.up1_cbr(sppf)
        up1 = self.up1(up1_cbr)
        combine_1 = torch.cat([up1, cbr1_c3], 1)
        up1_c3_0 = self.up1_c3_0(combine_1)
        up1_c3 = None
        for i, layer in enumerate(self.up1_c3):
            if i==0:
                up1_c3 = layer(up1_c3_0)
            else:
                up1_c3 = layer(up1_c3)
        if up1_c3 is None:
            up1_c3 = up1_c3_0
        
        up2_cbr = self.up2_cbr(up1_c3)
        up2 = self.up2(up2_cbr)
        combine_2 = torch.cat([up2, output_encoder], 1)
        up2_c3_0 = self.up2_c3_0(combine_2)
        up2_c3 = None
        for i, layer in enumerate(self.up2_c3):
            if i==0:
                up2_c3 = layer(up2_c3_0)
            else:
                up2_c3 = layer(up2_c3)
        if up2_c3 is None:
            up2_c3 = up2_c3_0
        
        down1_cbr = self.down1_cbr(up2_c3)
        combine_3 = torch.cat([down1_cbr, up2_cbr], 1)
        for i, layer in enumerate(self.down1_c3):
            if i==0:
                down1_c3 = layer(combine_3)
            else:
                down1_c3 = layer(down1_c3)
        
        down2_cbr = self.down2_cbr(down1_c3)
        combine_4 = torch.cat([down2_cbr, up1_cbr], 1)
        down2_c3_0 = self.down2_c3_0(combine_4)
        down2_c3 = None
        for i, layer in enumerate(self.down2_c3):
            if i==0:
                down2_c3 = layer(down2_c3_0)
            else:
                down2_c3 = layer(down2_c3)
        if down2_c3 is None:
            down2_c3 = down2_c3_0
        
        out_de = self.detect([up2_c3, down1_c3, down2_c3])

        return out_da, out_de
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
class Net3(nn.Module):
    def __init__(self, p = 1, q = 1, n = 3, planes = 4, classes = 10, name = 'n'):
        super(Net3, self).__init__()
        self.name = name
        
        self.dualbranch_encoder = ESP_Encoder(p = p, q = q, planes = planes)
        
        ### CAAM ###
        self.caam = CAAM(feat_in=planes * 4, num_classes= planes * 4, bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        
        ### Segmentation Task ###
        self.up_1_da = UpConvBlock(planes * 4, planes * 2) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(planes * 2, planes) # out: Hx2, Wx2
        self.out_da = UpConvBlock(planes, 2, last=True)  

        ### Object Detection Task ###
        self.cbr1 = Conv(planes * 4, planes * 8, 3, 2)
        self.cbr1_c3 = nn.ModuleList()
        for i in range(0, n):
            self.cbr1_c3.append(C3k2(planes * 8, planes * 8, shortcut=False))
        
        self.cbr2 = Convv5(planes * 8, planes * 16, 3, 2)
        self.cbr2_c3 = nn.ModuleList()
        for i in range(0, n):
            self.cbr2_c3.append(A2C2f(planes * 16, planes * 16, a2 = True, area = 4))
        
        self.sppf = A2C2f(planes * 16, planes * 16, a2 = True, area = 1)
        
        self.up1_cbr = Conv(planes * 16, planes * 8, 1, 1)
        self.up1 = nn.Upsample(scale_factor = 2, mode = "nearest")
        self.up1_c3_0 = C3k2(planes * 16, planes * 8, c3k=True)
        self.up1_c3 = nn.ModuleList()
        for i in range(1, n):
            self.up1_c3.append(C3k2(planes * 8, planes * 8, c3k=True))
        
        self.up2_cbr = Conv(planes * 8, planes * 4, 1, 1)
        self.up2 = nn.Upsample(scale_factor = 2, mode="nearest")
        self.up2_c3_0 = C3k2(planes * 8, planes * 4, c3k=True)
        self.up2_c3 = nn.ModuleList()
        for i in range(1, n):
            self.up2_c3.append(C3k2(planes * 4, planes * 4, c3k=True))
        
        self.down1_cbr = Conv(planes * 4, planes * 4, 3, 2)
        self.down1_c3 = nn.ModuleList()
        for i in range(0, n):
            self.down1_c3.append(C3k2(planes * 8, planes * 8, c3k=True))
        
        self.down2_cbr = Conv(planes * 8, planes * 8, 3, 2)
        self.down2_c3_0 = C3k2(planes * 16, planes * 32, c3k=True)
        self.down2_c3 = nn.ModuleList()
        for i in range(1, n):
            self.down2_c3.append(C3k2(planes * 32, planes * 32, c3k=True))    
        

        self.detect = Detect(classes, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [planes * 4, planes * 8, planes * 32])
        if isinstance(self.detect, Detect):
            s = 128  # 2x min stride
            self.eval()
            self.detect.train()
            with torch.no_grad():
                _,model_out = self.forward(torch.zeros(1, 3, s, s))
                detects= model_out
                self.detect.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            self.detect.anchors /= self.detect.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(self.detect)
            self.stride = self.detect.stride
            self._initialize_biases()
            self.train()
        
        initialize_weights(self)

    def forward(self, x):
        output_encoder, inp1, inp2 = self.dualbranch_encoder(x)
        # print('Encoder Delay: ', time.time()-start)

        spp = self.caam(output_encoder)

        ### Segmentation Task ###
        out_da = self.up_1_da(spp, inp2)
        out_da = self.up_2_da(out_da, inp1)
        out_da = self.out_da(out_da)

        ### Object Detection Task ###
        cbr1 = self.cbr1(output_encoder)
        for i, layer in enumerate(self.cbr1_c3):
            if i==0:
                cbr1_c3 = layer(cbr1)
            else:
                cbr1_c3 = layer(cbr1_c3)
        
        cbr2 = self.cbr2(cbr1_c3)
        for i, layer in enumerate(self.cbr2_c3):
            if i==0:
                cbr2_c3 = layer(cbr2)
            else:
                cbr2_c3 = layer(cbr2_c3)
        
        sppf = self.sppf(cbr2_c3)
        
        up1_cbr = self.up1_cbr(sppf)
        up1 = self.up1(up1_cbr)
        combine_1 = torch.cat([up1, cbr1_c3], 1)
        up1_c3_0 = self.up1_c3_0(combine_1)
        up1_c3 = None
        for i, layer in enumerate(self.up1_c3):
            if i==0:
                up1_c3 = layer(up1_c3_0)
            else:
                up1_c3 = layer(up1_c3)
        if up1_c3 is None:
            up1_c3 = up1_c3_0
        
        up2_cbr = self.up2_cbr(up1_c3)
        up2 = self.up2(up2_cbr)
        combine_2 = torch.cat([up2, output_encoder], 1)
        up2_c3_0 = self.up2_c3_0(combine_2)
        up2_c3 = None
        for i, layer in enumerate(self.up2_c3):
            if i==0:
                up2_c3 = layer(up2_c3_0)
            else:
                up2_c3 = layer(up2_c3)
        if up2_c3 is None:
            up2_c3 = up2_c3_0
        
        down1_cbr = self.down1_cbr(up2_c3)
        combine_3 = torch.cat([down1_cbr, up2_cbr], 1)
        for i, layer in enumerate(self.down1_c3):
            if i==0:
                down1_c3 = layer(combine_3)
            else:
                down1_c3 = layer(down1_c3)
        
        down2_cbr = self.down2_cbr(down1_c3)
        combine_4 = torch.cat([down2_cbr, up1_cbr], 1)
        down2_c3_0 = self.down2_c3_0(combine_4)
        down2_c3 = None
        for i, layer in enumerate(self.down2_c3):
            if i==0:
                down2_c3 = layer(down2_c3_0)
            else:
                down2_c3 = layer(down2_c3)
        if down2_c3 is None:
            down2_c3 = down2_c3_0
        
        out_de = self.detect([up2_c3, down1_c3, down2_c3])

        return out_da, out_de
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

class Net4(nn.Module):
    def __init__(self, p = 1, q = 1, n = 3, planes = 4, classes = 10, name = 'n'):
        super(Net4, self).__init__()
        self.name = name
        
        self.dualbranch_encoder = ESP_Encoder(p = p, q = q, planes = planes)
        
        ### CAAM ###
        self.caam = CAAM(feat_in=planes * 4, num_classes= planes * 4, bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        
        ### Segmentation Task ###
        self.up_1_da = UpConvBlock(planes * 4, planes * 2) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(planes * 2, planes) # out: Hx2, Wx2
        self.out_da = UpConvBlock(planes, 2, last=True)  

        ### Object Detection Task ###
        self.cbr1 = Conv(planes * 4, planes * 8, 3, 2)
        self.cbr1_c3 = nn.ModuleList()
        for i in range(0, n):
            self.cbr1_c3.append(C2f(planes * 8, planes * 8, shortcut=True))
        
        self.cbr2 = Conv(planes * 8, planes * 16, 3, 2)
        self.cbr2_c3 = nn.ModuleList()
        for i in range(0, n):
            self.cbr2_c3.append(C2f(planes * 16, planes * 16, shortcut=True))
        
        self.sppf = SPPF(planes * 16, planes * 16, 5)
        
        self.up1_cbr = Conv(planes * 16, planes * 8, 1, 1)
        self.up1 = nn.Upsample(scale_factor = 2, mode = "nearest")
        self.up1_c3_0 = C2f(planes * 16, planes * 8)
        self.up1_c3 = nn.ModuleList()
        for i in range(1, n):
            self.up1_c3.append(C2f(planes * 8, planes * 8))
        
        self.up2_cbr = Conv(planes * 8, planes * 4, 1, 1)
        self.up2 = nn.Upsample(scale_factor = 2, mode="nearest")
        self.up2_c3_0 = C2f(planes * 8, planes * 4)
        self.up2_c3 = nn.ModuleList()
        for i in range(1, n):
            self.up2_c3.append(C2f(planes * 4, planes * 4))
        
        self.down1_cbr = Conv(planes * 4, planes * 4, 3, 2)
        self.down1_c3 = nn.ModuleList()
        for i in range(0, n):
            self.down1_c3.append(C2f(planes * 8, planes * 8))
        
        self.down2_cbr = Conv(planes * 8, planes * 8, 3, 2)
        self.down2_c3_0 = C2f(planes * 16, planes * 32)
        self.down2_c3 = nn.ModuleList()
        for i in range(1, n):
            self.down2_c3.append(C2f(planes * 32, planes * 32))    
        

        self.detect = Detect(classes, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [planes * 4, planes * 8, planes * 32])
        if isinstance(self.detect, Detect):
            s = 128  # 2x min stride
            self.eval()
            self.detect.train()
            with torch.no_grad():
                _,model_out = self.forward(torch.zeros(1, 3, s, s))
                detects= model_out
                self.detect.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            self.detect.anchors /= self.detect.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(self.detect)
            self.stride = self.detect.stride
            self._initialize_biases()
            self.train()
        
        initialize_weights(self)

    def forward(self, x):
        output_encoder, inp1, inp2 = self.dualbranch_encoder(x)
        # print('Encoder Delay: ', time.time()-start)

        spp = self.caam(output_encoder)

        ### Segmentation Task ###
        out_da = self.up_1_da(spp, inp2)
        out_da = self.up_2_da(out_da, inp1)
        out_da = self.out_da(out_da)

        ### Object Detection Task ###
        cbr1 = self.cbr1(output_encoder)
        for i, layer in enumerate(self.cbr1_c3):
            if i==0:
                cbr1_c3 = layer(cbr1)
            else:
                cbr1_c3 = layer(cbr1_c3)
        
        cbr2 = self.cbr2(cbr1_c3)
        for i, layer in enumerate(self.cbr2_c3):
            if i==0:
                cbr2_c3 = layer(cbr2)
            else:
                cbr2_c3 = layer(cbr2_c3)
        
        sppf = self.sppf(cbr2_c3)
        
        up1_cbr = self.up1_cbr(sppf)
        up1 = self.up1(up1_cbr)
        combine_1 = torch.cat([up1, cbr1_c3], 1)
        up1_c3_0 = self.up1_c3_0(combine_1)
        up1_c3 = None
        for i, layer in enumerate(self.up1_c3):
            if i==0:
                up1_c3 = layer(up1_c3_0)
            else:
                up1_c3 = layer(up1_c3)
        if up1_c3 is None:
            up1_c3 = up1_c3_0
        
        up2_cbr = self.up2_cbr(up1_c3)
        up2 = self.up2(up2_cbr)
        combine_2 = torch.cat([up2, output_encoder], 1)
        up2_c3_0 = self.up2_c3_0(combine_2)
        up2_c3 = None
        for i, layer in enumerate(self.up2_c3):
            if i==0:
                up2_c3 = layer(up2_c3_0)
            else:
                up2_c3 = layer(up2_c3)
        if up2_c3 is None:
            up2_c3 = up2_c3_0
        
        down1_cbr = self.down1_cbr(up2_c3)
        combine_3 = torch.cat([down1_cbr, up2_cbr], 1)
        for i, layer in enumerate(self.down1_c3):
            if i==0:
                down1_c3 = layer(combine_3)
            else:
                down1_c3 = layer(down1_c3)
        
        down2_cbr = self.down2_cbr(down1_c3)
        combine_4 = torch.cat([down2_cbr, up1_cbr], 1)
        down2_c3_0 = self.down2_c3_0(combine_4)
        down2_c3 = None
        for i, layer in enumerate(self.down2_c3):
            if i==0:
                down2_c3 = layer(down2_c3_0)
            else:
                down2_c3 = layer(down2_c3)
        if down2_c3 is None:
            down2_c3 = down2_c3_0
        
        out_de = self.detect([up2_c3, down1_c3, down2_c3])

        return out_da, out_de
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
import time
def time_c():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    from thop import profile
    
    import torch
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--type', default="nano", help='')
    parser.add_argument('--is320', action='store_true')
    parser.add_argument('--seda', action='store_true', help='sigle encoder for Drivable Segmentation')
    parser.add_argument('--sell', action='store_true', help='sigle encoder for Lane Segmentation')
    args = parser.parse_args()
    
    # for scale in ["nano","small","medium","large"]:
    # args.type = scale
    # print(scale)
    model = Net4(p = 1, q = 1, n = 1, planes = 4, classes = 14, name = 'n').cuda()
    cudnn.benchmark = True
    model.eval()
    example = torch.randn(1, 3, 384, 640).cuda()
    # model = torch.jit.trace(model, example)
    
    
    for i in range(50):
        model(example)
    st=time_c()
    for i in range(500):
        model(example)
    print(1/((time_c()-st)/500))
    
    print('Output Shape:', model(example)[0].shape, model(example)[1][0].shape)
    # print('Scale: {}, ImSize: {}x{}'.format(scale, size, size))
    flops, params = profile(model, inputs=(example, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.4f}'.format(flops / 1e9))
    print('Params : {:.4f} M'.format(params / 1e6))
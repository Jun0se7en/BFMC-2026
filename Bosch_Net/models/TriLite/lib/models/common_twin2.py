import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
import math


chanel_img = 3
sc_ch_dict = {
    "nano": {  'p': 1,
            'q': 1,
            'chanels' : [4,8, 16, 32, 64],
    },
    
    "small": {  'p': 2,
            'q': 3,
            'chanels' : [8,16, 32, 64, 128],
    },

    "medium": {  'p': 3,
            'q': 5,
            'chanels' : [16,32, 64, 128, 256],
    },

    "large": {  'p': 5,
            'q': 7,
            'chanels' : [32,64, 128, 256, 512],
    }
}

from torch.nn import Module, Conv2d, Parameter, Softmax
import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Khởi tạo FPN module.

        :param in_channels_list: Danh sách các số lượng channels của input feature maps (ví dụ: [256, 512, 1024]).
        :param out_channels: Số lượng channels của output feature maps.
        """
        super(FPN, self).__init__()
        
        # 1x1 convolutions để điều chỉnh số lượng channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # 3x3 convolutions để làm mịn các output
        self.output_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3)
            for _ in in_channels_list
        ])
    
    def forward(self, inputs):
        """
        Xử lý forward của FPN module.

        :param inputs: Danh sách các input feature maps (ví dụ: [P3, P4, P5]).
        :return: Danh sách các output feature maps đã qua FPN.
        """
        # Step 1: Điều chỉnh số lượng channels cho mỗi input feature map
        P5, P4, P3 = inputs
        M5 = self.lateral_convs[2](P5)
        M4 = self.lateral_convs[1](P4)
        M3 = self.lateral_convs[0](P3)
        
        # Step 2: Top-down pathway
        # Up-sample M5 và cộng với P4 đã qua 1x1 conv
        M4 = M4 + F.interpolate(M5, size=M4.shape[-2:], mode='bilinear', align_corners=False)
        # Up-sample M4 và cộng với P3 đã qua 1x1 conv
        M3 = M3 + F.interpolate(M4, size=M3.shape[-2:], mode='bilinear', align_corners=False)
        
        # Step 3: Output convolutions (3x3 conv)
        O5 = self.output_convs[2](M5)
        O4 = self.output_convs[1](M4)
        O3 = self.output_convs[0](M3)
        
        return [O3, O4, O5]
def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0,2,4,3,5,1).contiguous() # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B,-1,rH,rW,C) # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out

def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0,5,1,3,2,4).contiguous() # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W) # [B, C, H, W]
    return out

class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.PReLU(num_node)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out
class CAAM(nn.Module):
    """
    Class Activation Attention Module
    """
    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)
              
        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.PReLU(feat_in)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.PReLU(1)

    def forward(self, x):
        cam = self.conv_cam(x) # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam)) # [B, K, bin_num_h, bin_num_w]

        residual = x # [B, C, H, W]
        cam = patch_split(cam, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, K]
        x = patch_split(x, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH*rW, K) # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH*rW, C) # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B,K,-1).transpose(1,2).unsqueeze(3) # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x) * bin_confidence # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats) # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats) # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1) # [B, bin_num_h * bin_num_w, K, C]
        
        query = self.proj_query(x) # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        
        aff_map = torch.matmul(query, key.transpose(2, 3)) # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value) # [B, bin_num_h * bin_num_w, rH * rW, C]
        
        out = out.view(B, -1, rH, rW, value.shape[-1]) # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size) # [B, C, H, W]

        out = residual + self.conv_out(out)
        return out



class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize=3, stride=1, groups=1,dropout_rate=0.0):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        if self.dropout:
            output = self.dropout(output)
        return output

class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut=nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.depthwise = nn.Conv2d(nin, nin, kernel_size, stride, padding, dilation, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = CDilated(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        output = self.bn(combine)
        output = self.act(output)
        return output




class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        self.c1 = DepthwiseSeparableConv(nIn, n, 1, 1)
        self.d1 = DepthwiseSeparableConv(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = DepthwiseSeparableConv(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = DepthwiseSeparableConv(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = DepthwiseSeparableConv(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = DepthwiseSeparableConv(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class AvgPool(nn.Module):

    def __init__(self):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, input):
        print(input.shape,"1")
        input = self.pool(input)
        print(input.shape,"2")
        return input


class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self):
        super().__init__()
        chanel_img = 3
        model_cfg = sc_ch_dict["small"] 
        self.level1 = CBR(chanel_img, model_cfg['chanels'][0], stride = 2)
        # self.sample1 = InputProjectionA(1)
        # self.sample2 = InputProjectionA(2)

        self.b1 = CBR(model_cfg['chanels'][0] + chanel_img,model_cfg['chanels'][1])
        self.level2_0 = DownSamplerB(model_cfg['chanels'][1], model_cfg['chanels'][2])

        self.level2 = nn.ModuleList()
        for i in range(0, model_cfg['p']):
            self.level2.append(DilatedParllelResidualBlockB(model_cfg['chanels'][2] , model_cfg['chanels'][2]))
        self.b2 = CBR(model_cfg['chanels'][3] + chanel_img,model_cfg['chanels'][3] + chanel_img)

        self.level3_0 = DownSamplerB(model_cfg['chanels'][3] + chanel_img, model_cfg['chanels'][3])
        self.level3 = nn.ModuleList()
        for i in range(0, model_cfg['q']):
            self.level3.append(DilatedParllelResidualBlockB(model_cfg['chanels'][3] , model_cfg['chanels'][3]))
        # self.b3 = CBR(model_cfg['chanels'][4],model_cfg['chanels'][2])
        
    def forward(self, input_):
        input, inp1, inp2 = input_
        print(input.shape, inp1.shape, inp2.shape)
        output0 = self.level1(input)
        # inp1 = self.sample1(input)
        # inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        out_encoder=torch.cat([output2_0, output2], 1)
        # out_encoder = self.b3(output2_cat)
        
        return out_encoder

class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False,kernel_size = 3):
        super(UpConvBlock, self).__init__()
        self.last=last
        self.up_conv = UpSimpleBlock(in_channels, out_channels)
        if not last:
            self.conv1 = CBR(out_channels+sub_dim,out_channels,kernel_size)
        self.conv2 = CBR(out_channels,out_channels,kernel_size)

    def forward(self, x, ori_img=None):
        x = self.up_conv(x)
        if not self.last:
            x = torch.cat([x, ori_img], dim=1)
            x = self.conv1(x)
        x = self.conv2(x)
        return x



def init_params(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
        
# Example usage
class DetectHead(nn.Module):

    def __init__(self):
        super().__init__()
        model_cfg = sc_ch_dict["small"] 
        # self.down1 = InputProjectionA(1)
        # self.down2 = InputProjectionA(2)
        

        self.cbr1 = CBR(model_cfg['chanels'][4],model_cfg['chanels'][4], 3, 2)
        self.cbr2 = CBR(model_cfg['chanels'][4],model_cfg['chanels'][4], 3, 2)

        self.spp = SPP(model_cfg['chanels'][4],model_cfg['chanels'][4])
        self.fpn = FPN(in_channels_list=[model_cfg['chanels'][4],model_cfg['chanels'][4],model_cfg['chanels'][4]], out_channels=model_cfg['chanels'][3])


    def forward(self, output_encoder):

        out1 = self.cbr1(output_encoder)
        out2 = self.cbr2(out1)
        out2 = self.spp(out2)

        out1, out2, out3 = self.fpn([out2, out1, output_encoder])

        return [out1,out2,out3]

class CAAMC(nn.Module):
    def __init__(self):
        super().__init__()
        model_cfg = sc_ch_dict["small"] 
        self.caam = CAAM(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][4],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = CBR(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][2])

    def forward(self, out_encoder):
        
        out_caam = self.caam(out_encoder)
        out_caam=self.conv_caam(out_caam)
        return out_caam


# class SegmentHead(nn.Module):
#     def __init__(self):
#         super().__init__()
#         model_cfg = sc_ch_dict["small"] 
#         self.caam = CAAM(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][4],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
#         self.conv_caam = CBR(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][2])

#         self.up_1_da = UpConvBlock(sc_ch_dict["small"]['chanels'][2],sc_ch_dict["small"]['chanels'][1]) # out: Hx4, Wx4
#         self.up_2_da = UpConvBlock(sc_ch_dict["small"]['chanels'][1],sc_ch_dict["small"]['chanels'][0]) #out: Hx2, Wx2
 

#         self.up_1_ll = UpConvBlock(sc_ch_dict["small"]['chanels'][2],sc_ch_dict["small"]['chanels'][1]) # out: Hx4, Wx4
#         self.up_2_ll = UpConvBlock(sc_ch_dict["small"]['chanels'][1],sc_ch_dict["small"]['chanels'][0]) #out: Hx2, Wx2

#     def forward(self, out_encoder):

#         out_encoder,inp1,inp2 = out_encoder

#         out_caam = self.caam(out_encoder)
#         out_caam=self.conv_caam(out_caam)

#         out_da=self.up_1_da(out_caam,inp2)
#         out_da=self.up_2_da(out_da,inp1)

#         out_ll=self.up_1_ll(out_caam,inp2)
#         out_ll=self.up_2_ll(out_ll,inp1)

#         return out_da,out_ll

class SegHead(nn.Module):
    def __init__(self,):
        super().__init__()
        model_cfg = sc_ch_dict["small"] 
        self.up_1_da = UpConvBlock(sc_ch_dict["small"]['chanels'][2],sc_ch_dict["small"]['chanels'][1]) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(sc_ch_dict["small"]['chanels'][1],sc_ch_dict["small"]['chanels'][0]) #out: Hx2, Wx2

    def forward(self, input_):
        
        out_caam, inp1, inp2 = input_
        print(out_caam.shape, inp1.shape, inp2.shape,"-----")
        out_da=self.up_1_da(out_caam,inp2)
        out_da=self.up_2_da(out_da,inp1)
        print(out_da.shape,"++++++")
        return out_da

class SegLL(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        _,input = input
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

# class CAModule(nn.Module):
#     def __init__(self):
#         model_cfg = sc_ch_dict["small"] 
#         self.caam = CAAM(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][4],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
#         self.conv_caam = CBR(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][2])

#     def 

# class SegmentHeadOne(nn.Module):
#     def __init__(self):
#         super().__init__()
#         model_cfg = sc_ch_dict["small"] 
#         self.caam = CAAM(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][4],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
#         self.conv_caam = CBR(sc_ch_dict["small"]['chanels'][4],sc_ch_dict["small"]['chanels'][2])

#         self.up_1_da = UpConvBlock(sc_ch_dict["small"]['chanels'][2],sc_ch_dict["small"]['chanels'][1]) # out: Hx4, Wx4
#         self.up_2_da = UpConvBlock(sc_ch_dict["small"]['chanels'][1],sc_ch_dict["small"]['chanels'][0]) #out: Hx2, Wx2
 

#         self.up_1_ll = UpConvBlock(sc_ch_dict["small"]['chanels'][2],sc_ch_dict["small"]['chanels'][1]) # out: Hx4, Wx4
#         self.up_2_ll = UpConvBlock(sc_ch_dict["small"]['chanels'][1],sc_ch_dict["small"]['chanels'][0]) #out: Hx2, Wx2

#     def forward(self, out_encoder):

#         out_encoder,inp1,inp2 = out_encoder

#         out_caam = self.caam(out_encoder)
#         out_caam=self.conv_caam(out_caam)

#         out_da=self.up_1_da(out_caam,inp2)
#         out_da=self.up_2_da(out_da,inp1)

#         out_ll=self.up_1_ll(out_caam,inp2)
#         out_ll=self.up_2_ll(out_ll,inp1)

#         return out_da,out_ll



if __name__ == "__main__":
    from thop import profile
    from torch.utils.tensorboard import SummaryWriter
    model = TwinLiteNet()
    init_params(model)
    input_1 = torch.randn((1, 3, 384, 640))
    # input_2 = torch.randn((1, 256, 384//8, 640//8))
    out_det, out_da, out_ll = model(input_1)
    for i in range(len(out_det)):
        print(out_det[i].shape)
    print(out_da.shape, out_ll.shape)

    macs, params = profile(model, inputs=(input_1, ))
    print(macs, params)

    
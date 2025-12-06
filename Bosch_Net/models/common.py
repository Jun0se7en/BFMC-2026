import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False
BN_MOMENTUM = 0.1

##### TwinLiteNet #####

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DepthSeperabelConv2d(nn.Module):
    """
    DepthSeperable Convolution 2d with residual connection
    """

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, act=True):
        super(DepthSeperabelConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        )
        # self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=1, bias=False)
        # self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.pointwise = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        )
        self.downsample = downsample
        self.stride = stride
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        #residual = x

        out = self.depthwise(x)
        out = self.act(out)
        out = self.pointwise(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.act(out)

        return out



class SharpenConv(nn.Module):
    # SharpenConv convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SharpenConv, self).__init__()
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        kenel_weight = np.vstack([sobel_kernel]*c2*c1).reshape(c2,c1,3,3)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv.weight.data = torch.from_numpy(kenel_weight)
        self.conv.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super(Bottleneck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class BottleneckCSP(nn.Module):
#     # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(BottleneckCSP, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.LeakyReLU(0.1, inplace=True)
#         self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x

class Detect(nn.Module):
    """YOLOv5 Detect head for processing input tensors and generating detection outputs in object detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=False):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class PAN(nn.Module):
    def __init__(self, input_channel, output_channel):
        """
        Khởi tạo PAN với chỉ bottom-up pathway và lớp làm mịn output.

        :param channels: Số lượng channels của input và output feature maps (giống nhau).
        """
        super(PAN, self).__init__()

        # Convolutions để giảm chiều không gian (stride=2) trong bottom-up pathway
        self.bottom_up_convs = nn.ModuleList([
            DepthwiseSeparableConv(input_channel, output_channel, kernel_size=3, stride=2)
            for _ in range(2)  # Số tầng giảm từ P3 -> P5
        ])

        # Convolutions để làm mịn output feature maps
        self.smooth_convs = nn.ModuleList([
            DepthwiseSeparableConv(output_channel, output_channel, kernel_size=3, stride=1)
            for _ in range(3)  # Làm mịn cho B3, B4, B5
        ])

    def forward(self, inputs):
        """
        Xử lý forward cho bottom-up PAN.

        :param inputs: Danh sách các input feature maps từ FPN (giả định là [P3, P4, P5]).
        :return: Danh sách các output feature maps từ bottom-up PAN ([B3, B4, B5]).
        """
        P5, P4, P3 = inputs  # Input từ FPN hoặc tương đương

        # Bottom-up pathway
        B3 = P3  # Bắt đầu từ feature map nhỏ nhất (P3)
        # print(self.bottom_up_convs[0](B3).shape,P3.shape, P4.shape)
        B4 = self.bottom_up_convs[0](B3) + P4  # Downsample B3 và cộng với P4
        B5 = self.bottom_up_convs[1](B4) + P5  # Downsample B4 và cộng với P5

        # Làm mịn output feature maps
        B3 = self.smooth_convs[0](B3)
        B4 = self.smooth_convs[1](B4)
        B5 = self.smooth_convs[2](B5)

        return [B3, B4, B5]

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

class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, p = 1, q = 1, channels = 4):
        super().__init__()
        # model_cfg = model_cfg 
        self.level1 = CBR(3, channels, stride = 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = CBR(channels + 3, channels * 2)
        self.level2_0 = DownSamplerB(channels * 2, channels * 4)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(channels * 4 , channels * 4))
        self.b2 = CBR(channels * 8 + 3,channels * 8 + 3)

        self.level3_0 = DownSamplerB(channels * 8 + 3, channels * 8)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(channels * 8 , channels * 8))
        self.b3 = CBR(channels * 16, channels)
        
    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1    = self.sample1(input)
        inp2    = self.sample2(input)
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
        out_encoder= self.b3(out_encoder)
        
        return out_encoder,inp1,inp2

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
        
class DetectHead(nn.Module):

    def __init__(self, model_cfg):
        super().__init__()
        # model_cfg = model_cfg 
        self.down1 = InputProjectionA(1)
        self.down2 = InputProjectionA(2)
        
        self.cbr1 = CBR(model_cfg['chanels'][4],model_cfg['chanels'][4], 3, 2)
        self.cbr2 = CBR(model_cfg['chanels'][4],model_cfg['chanels'][4], 3, 2)

        self.spp = SPP(model_cfg['chanels'][4],model_cfg['chanels'][4])
        self.fpn = FPN(in_channels_list=[model_cfg['chanels'][4],model_cfg['chanels'][4],model_cfg['chanels'][4]], out_channels=model_cfg['chanels'][3])
        self.pan = PAN(input_channel=model_cfg['chanels'][3], output_channel=model_cfg['chanels'][3])

    def forward(self, output_encoder):

        output_encoder,_,_=output_encoder

        out1 = self.cbr1(output_encoder)
        out2 = self.cbr2(out1)
        out2 = self.spp(out2)

        out1, out2, out3 = self.fpn([out2, out1, output_encoder])

        out1, out2, out3 = self.pan([out3, out2, out1])

        return [out1,out2,out3]

class SegmentHead(nn.Module):
    def __init__(self, model_cfg):

        super().__init__()

        # model_cfg = model_cfg 
        self.caam = CAAM(model_cfg['chanels'][4],model_cfg['chanels'][4],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = CBR(model_cfg['chanels'][4],model_cfg['chanels'][2])

        self.up_1_da = UpConvBlock(model_cfg['chanels'][2],model_cfg['chanels'][1]) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(model_cfg['chanels'][1],model_cfg['chanels'][0]) #out: Hx2, Wx2
 
        self.up_1_ll = UpConvBlock(model_cfg['chanels'][2],model_cfg['chanels'][1]) # out: Hx4, Wx4
        self.up_2_ll = UpConvBlock(model_cfg['chanels'][1],model_cfg['chanels'][0]) #out: Hx2, Wx2

    def forward(self, out_encoder):

        out_encoder,inp1,inp2 = out_encoder

        out_caam = self.caam(out_encoder)
        out_caam=self.conv_caam(out_caam)

        out_da=self.up_1_da(out_caam,inp2)
        out_da=self.up_2_da(out_da,inp1)

        out_ll=self.up_1_ll(out_caam,inp2)
        out_ll=self.up_2_ll(out_ll,inp1)

        return out_da,out_ll

##### DSNet #####

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = True
class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.PReLU(inplanes)  # 激活函数
        self.conv = nn.Conv2d(inplanes, planes * upscale_factor ** 2, kernel_size=3, padding=1)  # 卷积层
        self.bn = nn.BatchNorm2d(planes * upscale_factor ** 2)  # 批量归一化
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # 像素洗牌操作用于上采样
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)  # 通过PixelShuffle进行上采样
        x = self.conv2(x)
        return x

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation==1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, dilation=dilation,  padding=dilation, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.PReLU(out_planes)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out
    
class Conv1X1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, dilation=1):
        super(Conv1X1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.PReLU(out_planes)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

# 一个快速聚拢感受野的方法，改编自STDC
class MFACB(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[2,2,2]):
        super(MFACB, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        # self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        # self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        # self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.conv_list.append(CDilated(in_planes, inter_planes, kSize=3, stride=stride, d=dilation[0]))
        self.conv_list.append(CDilated(inter_planes, inter_planes, kSize=3, stride=stride, d=dilation[1]))
        self.conv_list.append(CDilated(inter_planes, inter_planes, kSize=3, stride=stride, d=dilation[2]))
        # self.process1 = nn.Sequential(
        #     nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
        #     BatchNorm2d(out_planes, momentum=bn_mom),
        #     nn.PReLU(out_planes)
        # )
        self.process1 = CBR(in_planes, out_planes, 1)
        # self.process2 = nn.Sequential(
        #     nn.Conv2d(inter_planes *3, out_planes, kernel_size=1, padding=0, bias=False ),
        #     BatchNorm2d(out_planes, momentum=bn_mom),
        #     nn.PReLU(out_planes)
        # )
        self.process2 = CBR(inter_planes *3, out_planes, 1)

    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        # out1 = self.conv_list[0](x)
        for idx in range(3):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        return self.process2(out) + out1    
    
class SPASPP(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[6,12,18,24]):
        super(SPASPP, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        # self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        # self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        # self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        # self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[3]))
        self.conv_list.append(CDilated(in_planes, inter_planes, kSize=3, stride=stride, d=dilation[0]))
        self.conv_list.append(CDilated(inter_planes, inter_planes, kSize=3, stride=stride, d=dilation[1]))
        self.conv_list.append(CDilated(inter_planes, inter_planes, kSize=3, stride=stride, d=dilation[2]))
        self.conv_list.append(CDilated(inter_planes, inter_planes, kSize=3, stride=stride, d=dilation[3]))
        self.pooling = ASPPPooling(in_planes, inter_planes)
        self.process1 = CBR(in_planes,out_planes, 1)
        # self.process1 = nn.Sequential(
        #     nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
        #     BatchNorm2d(out_planes, momentum=bn_mom),
        #     nn.PReLU(out_planes)
        # )
        # self.process2 = nn.Sequential(
        #     nn.Conv2d(inter_planes*5, out_planes, kernel_size=1, padding=0, bias=False ),
        #     BatchNorm2d(out_planes, momentum=bn_mom),
        #     nn.PReLU(out_planes)
        # )
        self.process2 = CBR(inter_planes*5,out_planes, 1)
        # self.process3 = nn.Sequential(
        #     nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
        #     BatchNorm2d(out_planes, momentum=bn_mom),
        #     nn.PReLU(out_planes)
        # )
        self.process3 = CBR(out_planes,out_planes, 3)
    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        # print(out1.shape)
        out2 = self.pooling(x)
        # print(out2.shape)
        # out1 = self.conv_list[0](x)
        for idx in range(4):
            out = self.conv_list[idx](out)
            out_list.append(out)
        # print(out.shape)
        out = torch.cat(out_list, dim=1)
        out = torch.cat((out,out2), dim=1)
        return self.process3(self.process2(out) + out1)
    
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False,dilation=1):
        super(BasicBlock, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
        #                        padding=dilation, dilation=dilation, bias=False)
        # self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        # self.relu = nn.PReLU(planes)
        self.c1 = CDilated(inplanes, planes, kSize=3, stride=stride, d=dilation)
        self.br1 = BR(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
        #                        padding=dilation, dilation=dilation, bias=False, )
        # self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.c2 = CDilated(planes, planes, kSize=3, stride=stride, d=dilation)
        self.br2 = BR(planes)
        self.bn2 = BatchNorm2d(planes, eps=1e-03)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        out = self.c1(x)
        out = self.br1(out)

        # out = self.conv2(out)
        # out = self.bn2(out)
        out = self.c2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            # return out
            return self.bn2(out)
        else:
            # return self.relu(out)
            return self.br2(out)

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True, dilation=1):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        # self.relu1 = nn.PReLU(planes)
        self.cbr1 = CBR(inplanes, planes, kSize=1, stride=stride)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=dilation, dilation=dilation, bias=False)
        # self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        # self.relu2 = nn.PReLU(planes)
        self.c2 = CDilated(planes, planes, kSize=3, stride=stride, d=dilation)
        self.br2 = BR(planes)
        # self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
        #                        bias=False)
        # self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        # self.relu3 = nn.PReLU(planes * self.expansion)
        self.cbr3 = CBR(planes, planes * self.expansion, kSize=1, stride=stride)
        self.cb3 = CB(planes, planes * self.expansion, kSize=1, stride=stride)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu1(out)
        out = self.cbr1(x)

        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu2(out)
        out = self.c2(out)
        out = self.br2(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.no_relu:
            # return out
            return self.cb3(out)
        else:
            # return self.relu(out)
            return self.cbr3(out)


class segmentheadCARAFE(nn.Module):

    def __init__(self, interplanes, outplanes, scale_factor=8, cp_rate=4):
        super(segmentheadCARAFE, self).__init__()
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.carafe = CARAFE(inC=interplanes, outC=outplanes, cp_rate=cp_rate, up_factor=scale_factor)
        self.scale_factor = scale_factor

    def forward(self, x):
        out = self.carafe(self.relu(self.bn2(x)))

        return out

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out
    
class segmenthead_c(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_c, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(x))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out
class segmenthead_drop(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_drop, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)

        self.drop = nn.Dropout(0.5)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.drop(self.relu(self.bn2(x)))
        out = self.conv2(x)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out

class segmenthead_l(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        self.bn3 = BatchNorm2d(outplanes, momentum=bn_mom)

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        out = self.bn3(out)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out




class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # nn.ReLU(inplace=True),
            CBR(channels, inter_channels, kSize=1, stride=1),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            CB(inter_channels, channels, kSize=1, stride=1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # nn.ReLU(inplace=True),
            CBR(channels, inter_channels, kSize=1, stride=1),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            CB(inter_channels, channels, kSize=1, stride=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

# MSAF
class Muti_AFF(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(Muti_AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # # nn.ReLU(inplace=True),
            # nn.PReLU(inter_channels),
            CBR(channels, inter_channels, kSize=1, stride=1),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            CB(inter_channels, channels, kSize=1, stride=1),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # nn.PReLU(inter_channels),
            CBR(channels, inter_channels, kSize=1, stride=1),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels)
            CB(inter_channels, channels, kSize=1, stride=1),
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # nn.PReLU(inter_channels),
            CBR(channels, inter_channels, kSize=1, stride=1),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels)
            CB(inter_channels, channels, kSize=1, stride=1),
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # nn.PReLU(inter_channels),
            CBR(channels, inter_channels, kSize=1, stride=1),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels)
            CB(inter_channels, channels, kSize=1, stride=1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            # nn.PReLU(inter_channels),
            CBR(channels, inter_channels, kSize=1, stride=1),
            # nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
            CB(inter_channels, channels, kSize=1, stride=1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)
        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


# MSAF_small
class MSAF_small(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(MSAF_small, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # self.context3 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((16, 16)),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels)
        # )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        # c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        # c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
        
# MSA
class MSA(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(MSA, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x 
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei
        return xo

# MSA_small
class MSA_small(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(MSA_small, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # self.context3 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((16, 16)),
        #     nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(inter_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channels)
        # )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x 
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        # c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        # c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei
        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


    

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18, 24], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


    

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
    
    
class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2, cp_rate=4, ifBN=False):
        super(CARAFE, self).__init__()
        self.ifBN = ifBN
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // cp_rate, 1)
        self.encoder = nn.Conv2d(inC // cp_rate, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)
        self.bn = nn.BatchNorm2d(outC)
    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        if self.ifBN:
            out_tensor = self.bn(out_tensor)
        return out_tensor


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class BottleneckC3k(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckC3k(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(BottleneckC3k(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BottleneckC3k(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else BottleneckC3k(self.c, self.c, shortcut, g) for _ in range(n)
        )

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=3
        )

        
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        max_attn = attn.max(dim=-1, keepdim=True).values 
        exp_attn = torch.exp(attn - max_attn)
        attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
        x = (v @ attn.transpose(-2, -1))
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = x + self.pe(v)
        x = self.proj(x)
        return x
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


### YOLOv5 ###
def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Convv5(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))
class Bottleneckv5(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Convv5(c1, c_, 1, 1)
        self.cv2 = Convv5(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Convv5(c1, c_, 1, 1)
        self.cv2 = Convv5(c1, c_, 1, 1)
        self.cv3 = Convv5(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneckv5(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SPP(nn.Module):
    """Implements Spatial Pyramid Pooling (SPP) for feature extraction, ref: https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input channels), c2 (output channels), k (kernel sizes)."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Convv5(c1, c_, 1, 1)
        self.cv2 = Convv5(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv5 models."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Convv5(c1, c_, 1, 1)
        self.cv2 = Convv5(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
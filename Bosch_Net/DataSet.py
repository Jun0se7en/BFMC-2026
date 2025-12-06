import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math
from PIL import Image
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
import albumentations as A

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def letterbox(combination, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """Resize the input image and automatically padding to suitable shape :https://zhuanlan.zhihu.com/p/172121380"""
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    img, gray = combination
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        gray = cv2.resize(gray, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)  # add border
    # print(img.shape)
    
    combination = (img, gray)
    return combination, ratio, (dw, dh)

def letterbox_for_image(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def RandomBilateralBlur(img, sigma_bila_low = 0.05, sigma_bila_high=1.0):
    """
    Apply Bilateral Filtering

    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = random.uniform(sigma_bila_low, sigma_bila_high)
    blurred_img = denoise_bilateral(np.array(img_rgb), sigma_spatial=sigma, channel_axis = 2)
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


    
def RandomGaussianBlur(img, sigma_gaus_a = 1.15, sigma_gaus_b=0.15):
    """
    Apply Gaussian Blur
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sigma = sigma_gaus_b + random.random() * sigma_gaus_a
    blurred_img = gaussian(np.array(img_rgb), sigma=sigma, channel_axis = 2)
    blurred_img *= 255
    blurred_img_rgb = Image.fromarray(blurred_img.astype(np.uint8))
    blurred_img_bgr = cv2.cvtColor(np.array(blurred_img_rgb), cv2.COLOR_RGB2BGR)
    return blurred_img_bgr


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def _box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

def random_perspective(combination, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img, gray = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = _box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    combination = (img, gray)
    return combination, targets


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, hyp, valid=False, transform = None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.transform = transform
        self.degrees = hyp["degrees"]
        self.translate = hyp["translate"]
        self.scale = hyp["scale"]
        self.shear = hyp["shear"]
        self.hgain = hyp["hgain"]
        self.sgain = hyp["sgain"]
        self.vgain = hyp["vgain"]
        self.Random_Crop = A.RandomCrop(width=hyp["width_crop"], height=hyp["height_crop"])

        self.prob_perspective = hyp["prob_perspective"]
        self.prob_flip = hyp["prob_flip"]
        self.prob_hsv = hyp["prob_hsv"]
        self.prob_bilateral = hyp["prob_bilateral"]
        self.prob_gaussian = hyp["prob_gaussian"]
        self.prob_crop = hyp["prob_crop"]
        
        self.Tensor = transforms.ToTensor()
        self.valid=valid
        if valid:
            # self.root='/home/ceec/huycq/TwinVast_1/bdd100k/images/val'
            self.root = os.path.join(data_dir, 'val/images/')
            self.names=os.listdir(self.root)
        else:
            # self.root='/home/ceec/huycq/TwinVast_1/bdd100k/images/train'
            self.root = os.path.join(data_dir, 'train/images/')
            self.names=os.listdir(self.root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_=640
        H_=384
        image_name=os.path.join(self.root,self.names[idx])

        if not self.valid:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        else:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        
        label1 = cv2.imread(os.path.splitext(image_name.replace("images","segment"))[0] + '.png', 0)

        resized_shape = [W_, H_]
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = image.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
            label1 = cv2.resize(label1, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = image.shape[:2]
        
        (image, label1), ratio, pad = letterbox((image, label1), resized_shape, auto=True, scaleup=not self.valid)
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        with open(os.path.splitext(image_name.replace("images","detection"))[0] + '.txt', "r", encoding="utf-8") as file:
            lines = file.read().splitlines()  # Tạo list, mỗi phần tử là 1 dòng

        det_data = []
        for line in lines:
            det_data.append(list(map(float, line.split(" "))))
            det_data[-1][0] = int(det_data[-1][0])

        det_label = np.array(det_data)
        labels=[]
        if len(det_label) > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
        
        if not self.valid:
            if random.random() < self.prob_perspective:
                combination = (image, label1)
                (image, label1), labels= random_perspective(
                    combination=combination,
                    targets=labels,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear
                )
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= image.shape[0]  # height
                labels[:, [1, 3]] /= image.shape[1]  # width

            # if random.random() < self.prob_flip:
            #     image = np.fliplr(image)
            #     label1 = np.fliplr(label1)
            #     if len(labels):
            #         labels[:, 1] = 1 - labels[:, 1]
            
            if random.random() < self.prob_bilateral:
                image = RandomBilateralBlur(image)
            if random.random() < self.prob_gaussian:
                image = RandomGaussianBlur(image)
            # if random.random() < 0.5:
            #     image = np.flipud(image)
            #     label1 = np.flipud(label1)
            #     if len(labels):
            #         labels[:, 2] = 1 - labels[:, 2]

        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= image.shape[0]  # height
                labels[:, [1, 3]] /= image.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)


        _,seg_b1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY_INV)
        _,seg1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg_b1 = self.Tensor(seg_b1)
        seg_da = torch.stack((seg_b1[0], seg1[0]),0)
        
        
        image = np.array(image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        target = [seg_da, labels_out]
       
        return torch.from_numpy(image), target, image_name, shapes
    
    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg = [], []
        for i, l in enumerate(label):
            l_seg, l_det = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
        return torch.stack(img, 0), [torch.stack(label_seg, 0), torch.vstack(label_det)], paths, shapes

class Dataset320(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, data_dir, hyp, valid=False, transform = None):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.transform = transform
        self.degrees = hyp["degrees"]
        self.translate = hyp["translate"]
        self.scale = hyp["scale"]
        self.shear = hyp["shear"]
        self.hgain = hyp["hgain"]
        self.sgain = hyp["sgain"]
        self.vgain = hyp["vgain"]
        self.Random_Crop = A.RandomCrop(width=hyp["width_crop"], height=hyp["height_crop"])

        self.prob_perspective = hyp["prob_perspective"]
        self.prob_flip = hyp["prob_flip"]
        self.prob_bilateral = hyp["prob_bilateral"]
        self.prob_gaussian = hyp["prob_gaussian"]
        self.prob_crop = hyp["prob_crop"]
        
        self.Tensor = transforms.ToTensor()
        self.valid=valid
        if valid:
            # self.root='../bdd100k/images/val'
            self.root = os.path.join(data_dir, 'val/images/')
            self.names=os.listdir(self.root)
        else:
            # self.root='../bdd100k/images/train'
            self.root = os.path.join(data_dir, 'train/images/')
            self.names=os.listdir(self.root)


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_=320
        H_=192
        image_name=os.path.join(self.root,self.names[idx])

        if not self.valid:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        else:
            image = cv2.imread(os.path.join(self.root,self.names[idx]))
        
        label1 = cv2.imread(image_name.replace("images","segment").replace("jpg","png"), 0)
        label2 = cv2.imread(image_name.replace("images","lane").replace("jpg","png"), 0)
        


        if not self.valid:
            if random.random() < self.prob_perspective:
                combination = (image, label1, label2)
                (image, label1, label2)= random_perspective(
                    combination=combination,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    shear=self.shear
                )
            if random.random() < self.prob_hsv:
                augment_hsv(image, self.hgain, self.sgain, self.vgain)
            if random.random() < self.prob_flip:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)
            
            if random.random() < self.prob_bilateral:
                image = RandomBilateralBlur(image)
            if random.random() < self.prob_gaussian:
                image = RandomGaussianBlur(image)
            if random.random() < self.prob_crop:
                masks = np.stack([label1, label2],axis=2)
                transformed = self.Random_Crop(image=image, mask=masks)
                image = transformed['image']
                labels = transformed['mask']
                label1 = labels[:,:,0]
                label2 = labels[:,:,1]


            image = letterbox(image, (H_, W_))

            label1 = cv2.resize(label1, (W_, 180))
            label2 = cv2.resize(label2, (W_, 180))

        else:

            image = letterbox(image, (H_, W_))

            label1 = cv2.resize(label1, (W_*2, 360))
            label2 = cv2.resize(label2, (W_*2, 360))
        

        _,seg_b1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY_INV)
        _,seg_b2 = cv2.threshold(label2,1,255,cv2.THRESH_BINARY_INV)
        _,seg1 = cv2.threshold(label1,1,255,cv2.THRESH_BINARY)
        _,seg2 = cv2.threshold(label2,1,255,cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        seg_da = torch.stack((seg_b1[0], seg1[0]),0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]),0)
        image = np.array(image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)


       
        return image_name,torch.from_numpy(image),(seg_da,seg_ll)
    
    
    




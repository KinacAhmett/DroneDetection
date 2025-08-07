# This file contains modules common to various models

# Standard libraries
import math

# Scientific and tensor libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Project specific imports
from detect_wrapper.utils.datasets import letterbox
from detect_wrapper.utils.general import (
    non_max_suppression, make_divisible, scale_coords
)


def autopad(k, p=None):  # kernel, padding
    """
    Auto padding function that returns padding size based on kernel dimension.
    """
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    """
    Depthwise convolution using channel groups based on the GCD of channels.
    """
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()

    def forward(self, x):
        """
        Forward pass with batch normalization.
        """
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """
        Forward pass with fused operations for inference optimization.
        """
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    Standard bottleneck module with residual connection.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass with optional residual connection.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """
    Cross Stage Partial Bottleneck from https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        """
        Forward pass applying cross stage partial bottleneck operations.
        """
        y1 = self.cv3(self.m(self.cv1(x)))  # Process first branch
        y2 = self.cv2(x)                    # Process second branch
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    """
    Spatial pyramid pooling layer used in YOLOv3-SPP to capture multi-scale features.
    """
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """
        Apply spatial pyramid pooling at different kernel sizes.
        """
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    """
    Focus module that processes spatial information into channel-space for efficiency.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

    def forward(self, x):
        """
        Apply the focus operation on input tensor.
        """
        return self.conv(x)


class Concat(nn.Module):
    """
    Concatenate tensors along specified dimension with size adaptation.
    """
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """
        Resize all tensors to the smallest common size and concatenate them.
        """
        # Resize all tensors to the smallest size
        min_h = min([t.shape[2] for t in x])
        min_w = min([t.shape[3] for t in x])
        x = [F.interpolate(t, size=(min_h, min_w), mode='nearest') if (t.shape[2] != min_h or t.shape[3] != min_w) else t for t in x]
        for i, t in enumerate(x):
            print(f"Concat input {i}: {t.shape}")  # Print shape information for debugging
        return torch.cat(x, self.d)


class NMS(nn.Module):
    """
    Non-Maximum Suppression (NMS) module for filtering overlapping bounding boxes.
    """
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        """
        Apply Non-Maximum Suppression to model predictions.
        """
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    """
    Input-robust model wrapper handling cv2/np/PIL/torch inputs with preprocessing, inference and NMS.
    """
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model

    def forward(self, x, size=640, augment=False, profile=False):
        # supports inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   opencv:     x = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:        x = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:      x = np.zeros((720,1280,3))  # HWC
        #   torch:      x = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:   x = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(x, torch.Tensor):  # torch
            return self.model(x.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        if not isinstance(x, list):
            x = [x]
        shape0, shape1 = [], []  # image and inference shapes
        batch = range(len(x))  # batch size
        for i in batch:
            x[i] = np.array(x[i])[:, :, :3]  # up to 3 channels if png
            s = x[i].shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(x[i], new_shape=shape1, auto=False)[0] for i in batch]  # pad
        x = np.stack(x, 0) if batch[-1] else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        x = self.model(x, augment, profile)  # forward
        x = non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in batch:
            if x[i] is not None:
                x[i][:, :4] = scale_coords(shape1, x[i][:, :4], shape0[i])
        return x


class Flatten(nn.Module):
    """
    Flatten module used after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions.
    """
    @staticmethod
    def forward(x):
        """
        Flatten spatial dimensions to create a 2D tensor (batch_size, features).
        """
        return x.view(x.size(0), -1)


class Classify(nn.Module):
    """
    Classification head that converts feature maps x(b,c1,h,w) to classification outputs x(b,c2).
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)  # to x(b,c2,1,1)
        self.flat = Flatten()

    def forward(self, x):
        """
        Convert feature maps to classification output via adaptive pooling and convolution.
        """
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

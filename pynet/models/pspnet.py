# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
The Pyramid Scene Parsing Network.
"""

# Imports
import math
import logging
import collections
import torch
import torch.nn as nn
import torch.nn.functional as func
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import numpy as np


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("encoder", "segmenter"))
class PSPNet(nn.Module):
    """ Pyramid Scene Parsing Network.

    Reference: https://arxiv.org/pdf/1612.01105.pdf
    Code: https://github.com/cv-lee/BraTs
    """
    def __init__(self, n_classes=2, sizes=(1, 2, 3, 6), psp_size=2048,
                 deep_features_size=1024, backend="resnet34", drop_rate=0):
        """ Init class.

        Parameters
        ----------
        n_classes: int, default 2
            the number of features in the output segmentation map.
        """
        nn.Module.__init__(self)
        if backend == "resnet18":
            self.feats = resnet18(drop_rate)
        elif backend == "resnet34":
            self.feats = resnet34(drop_rate)
        elif backend == "resnet50":
            self.feats = resnet50(drop_rate)
        else:
            raise ValueError("Unsupported backend.")
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.up1 = PSPUpsample(1024, 256, drop_rate)
        self.up2 = PSPUpsample(256, 64, drop_rate)
        self.up3 = PSPUpsample(64, 64, drop_rate)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax())
        # self.classifier = nn.Sequential(
        #    nn.Linear(deep_features_size, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, n_classes))

    def forward(self, x):
        logger.debug("PSPNet...")
        self.debug("input", x)
        # f, class_f = self.feats(x)
        f, _ = self.feats(x)
        self.debug("feats", f)
        p = self.psp(f)
        self.debug("psp", p)
        p = self.up1(p)
        self.debug("up1", p)
        p = self.up2(p)
        self.debug("up2", p)
        p = self.up3(p)
        self.debug("up3", p)
        # auxiliary = func.adaptive_max_pool2d(
        #    input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))
        p = self.final(p)
        self.debug("final", p)
        # p = func.softmax(p, dim=1)
        logger.debug("Done.")
        return p  # , self.classifier(auxiliary)

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6),
                 drop_rate=0):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([
            self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features,
                                    kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            func.interpolate(input=stage(feats), size=(h, w), mode="bilinear",
                             align_corners=True) for stage in self.stages]
        priors += [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=drop_rate),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = func.interpolate(
            input=x, size=(h, w), mode="bilinear", align_corners=True)
        return self.conv(p)


class ResNet(nn.Module):
    """ Implementation of dilated ResNet with deep supervision. Downsampling
    is changed to 8x.
    """
    def __init__(self, block, layers=(3, 4, 23, 3), drop_rate=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilation=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=4, drop_rate=drop_rate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    drop_rate=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                nn.Dropout2d(drop_rate)
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        drop_rate=drop_rate)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                drop_rate=drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)
        return x, x_3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                 drop_rate=0):
        super(BasicBlock, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.dp(out) # Bottom or Here
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.dp(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                 drop_rate=0):
        super(Bottleneck, self).__init__()
        self.dp = nn.Dropout2d(p=drop_rate)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation,
            padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dp(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dp(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.dp(out)
        out = self.relu(out)
        return out


def resnet18(drop_rate=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], drop_rate=drop_rate)
    return model


def resnet34(drop_rate=0):
    model = ResNet(BasicBlock, [3, 4, 6, 3], drop_rate=drop_rate)
    return model


def resnet50(drop_rate=0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], drop_rate=drop_rate)
    return model


def pspnet_res18(drop_rate=0):
    return PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256,
                  backend="resnet18", drop_rate=drop_rate)


def pspnet_res34(drop_rate=0):
    return PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256,
                  backend="resnet34", drop_rate=drop_rate)


def pspnet_res50(drop_rate=0):
    return PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024,
                  backend="resnet50", drop_rate=drop_rate)

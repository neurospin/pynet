# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import collections
import torch
import torch.nn as nn
import torch.nn.functional as func


class UNet(nn.Module):
    """ UNet.

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1x1 3d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=1, depth=5, 
                 start_filts=64, up_mode="transpose", 
                 merge_mode="concat", batchnorm=False):
        """ Init class.

        Parameters
        ----------
        num_classes: int
            the number of features in the output segmentation map.
        in_channels: int, default 1
            number of channels in the input tensor.
        depth: int, default 5
            number of layers in the U-Net.
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        up_mode: string, default 'transpose'
            type of upconvolution. Choices: 'transpose' for transpose
            convolution or 'upsample' for nearest neighbour upsampling.
        merge_mode: str, defatul 'concat'
            the skip connections merging strategy.
        batchnorm: bool, default False
            normalize the inputs of the activation function.
        """
        # Inheritance
        super(UNet, self).__init__()

        # Check inputs
        if up_mode in ("transpose", "upsample"):
            self.up_mode = up_mode
        else:
            raise ValueError(
                "'{}' is not a valid mode for upsampling. Only 'transpose' "
                "and 'upsample' are allowed.".format(up_mode))
        if merge_mode in ("concat", "add"):
            self.merge_mode = merge_mode
        else:
            raise ValueError(
                "'{}' is not a valid mode for merging up and down paths. Only "
                "'concat' and 'add' are allowed.".format(up_mode))
        if self.up_mode == "upsample" and self.merge_mode == "add":
            raise ValueError(
                "up_mode 'upsample' is incompatible with merge_mode 'add' at "
                "the moment because it doesn't make sense to use nearest "
                "neighbour to reduce depth channels (by half).")

        # Declare class parameters
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down = []
        self.up = []

        # Create the encoder pathway
        for cnt in range(depth):
            in_channels = self.in_channels if cnt == 0 else out_channels
            out_channels = self.start_filts * (2**cnt)
            pooling = False if cnt == 0 else True
            self.down.append(
                Down(in_channels, out_channels, pooling=pooling,
                     batchnorm=batchnorm))

        # Create the decoder pathway
        # - careful! decoding only requires depth-1 blocks
        for cnt in range(depth - 1):
            in_channels = out_channels
            out_channels = in_channels // 2
            self.up.append(
                Up(in_channels, out_channels, up_mode=up_mode,
                   merge_mode=merge_mode, batchnorm=batchnorm))

        # Add the list of modules to current module
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

        # Get ouptut segmentation
        self.conv_final = Conv1x1x1(out_channels, self.num_classes)

        # Kernel initializer
        self.kernel_initializer()

    def kernel_initializer(self):
        for module in self.modules():
            self.init_weight(module)

    @staticmethod
    def init_weight(module):
        if isinstance(module, nn.Conv3d):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoder_outs = []
        for module in self.down:
            x = module(x)
            encoder_outs.append(x)
        encoder_outs = encoder_outs[:-1][::-1]
        for cnt, module in enumerate(self.up):
            x_up = encoder_outs[cnt]
            x = module(x, x_up)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss in your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, batchnorm=True):
        super(DoubleConv, self).__init__()
        if batchnorm:
            self.ops = nn.Sequential(collections.OrderedDict([
                ("conv1", nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)),
                ("norm1", nn.BatchNorm3d(out_channels)),
                ("leakyrelu1", nn.LeakyReLU()),
                ("conv2", nn.Conv3d(out_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)),
                ("norm2", nn.BatchNorm3d(out_channels)),
                ("leakyrelu2", nn.LeakyReLU())]))
        else:
            self.ops = nn.Sequential(collections.OrderedDict([
                ("conv1", nn.Conv3d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)),
                ("leakyrelu1", nn.LeakyReLU()),
                ("conv2", nn.Conv3d(out_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, bias=bias)),
                ("leakyrelu2", nn.LeakyReLU())]))

    def forward(self, x):
        x = self.ops(x)
        return x


def UpConv(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same as in_channels
        return nn.Sequential(collections.OrderedDict([
            ("up", nn.Upsample(mode="nearest", scale_factor=2)),
            ("conv1x", Conv1x1x1(in_channels, out_channels))]))


def Conv1x1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class Down(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A LeakyReLU activation and optionally a BatchNorm follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, batchnorm=True):
        super(Down, self).__init__()
        if pooling:
            self.ops = nn.Sequential(collections.OrderedDict([
                ("maxpool", nn.MaxPool3d(2)),
                ("doubleconv", DoubleConv(in_channels, out_channels, batchnorm=batchnorm))]))
        else:
            self.ops = nn.Sequential(collections.OrderedDict([
                ("doubleconv", DoubleConv(in_channels, out_channels, batchnorm=batchnorm))]))

    def forward(self, x):
        x = self.ops(x)
        return x


class Up(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A LeakyReLU activation and optionally a BatchNorm follows each convolution.
    """
    def __init__(self, in_channels, out_channels, merge_mode="concat",
                 up_mode="transpose", batchnorm=True):
        super(Up, self).__init__()
        self.merge_mode = merge_mode
        self.upconv = UpConv(in_channels, out_channels, mode=up_mode)
        self.doubleconv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x_down, x_up):
        x_down = self.upconv(x_down)
        if self.merge_mode == "concat":
            x = torch.cat((x_up, x_down), dim=1)
        else:
            x = x_up + x_down
        x = self.doubleconv(x)
        return x



# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
NvNet: combination of Vnet and VAE (variation auto-encoder).
"""

# Imports
import logging
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family="segmenter")
class NvNet(nn.Module):
    """ NvNet: combination of Vnet and VAE (variation auto-encoder).

    The variational auto-encoder branch reconstruct the input image jointly
    with segmentation in order to regularized the shared encoder.

    Reference: https://arxiv.org/pdf/1810.11654.pdf.
    Code: https://github.com/athon2/BraTS2018_NvNet.
    """
    def __init__(self, input_shape, in_channels, num_classes,
                 activation="relu", normalization="group_normalization",
                 mode="trilinear", with_vae=True):
        """ Init class.

        Parameters
        ----------
        input_shape: uplet
            the tensor data shape (X, Y, Z).
        in_channels: int
            number of channels in the input tensor.
        num_classes: int
            the number of features in the output segmentation map.
        activation: str, default 'relu'
            the activation function.
        normalization: str, default 'group_normalization'
            the normalization function.
        mode: str, default 'trilinear'
            the interpolation mode.
        with_vae: bool, default True
            enable/disable vae penalty.
        """
        # Inheritance
        nn.Module.__init__(self)

        # Check inputs
        if activation not in ("relu", "elu"):
            raise ValueError(
                "'{}' is not a valid activation. Only 'relu' "
                "and 'elu' are allowed.".format(activation))
        if normalization not in ("group_normalization"):
            raise ValueError(
                "'{}' is not a valid normalization. Only "
                "'group_normalization' is allowed.".format(normalization))
        if mode not in ("nearest", "linear", "bilinear", "bicubic",
                        "trilinear", "area"):
            raise ValueError(
                "'{}' is not a valid interpolation mode: see "
                "'torch.nn.functional.interpolate'for a list of allowed "
                "modes.".format(mode))

        # Declare class parameters
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.activation = activation
        self.normalization = normalization
        self.mode = mode
        self.with_vae = with_vae

        # Encoder Blocks: encoder parts uses ResNet blocks (two 3x3x3 conv,
        # group normlization (better than batch norm for small batch size),
        # and ReLU), followed by additive identity skip connection.
        # A Downsizing of 2 is performed with strided convolutions.
        # Features increase by two at each level.
        self.in_conv0 = DownSampling(
            in_channels=self.in_channels, out_channels=32, stride=1,
            kernel_size=3, dropout_rate=0.2, bias=True)
        self.en_block0 = EncoderBlock(
            in_channels=32, out_channels=32, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_down1 = DownSampling(
            in_channels=32, out_channels=64, stride=2, kernel_size=3)
        self.en_block1_0 = EncoderBlock(
            in_channels=64, out_channels=64, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_block1_1 = EncoderBlock(
            in_channels=64, out_channels=64, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_down2 = DownSampling(
            in_channels=64, out_channels=128, stride=2, kernel_size=3)
        self.en_block2_0 = EncoderBlock(
            in_channels=128, out_channels=128, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_block2_1 = EncoderBlock(
            in_channels=128, out_channels=128, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_down3 = DownSampling(
            in_channels=128, out_channels=256, stride=2, kernel_size=3)
        self.en_block3_0 = EncoderBlock(
            in_channels=256, out_channels=256, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_block3_1 = EncoderBlock(
            in_channels=256, out_channels=256, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_block3_2 = EncoderBlock(
            in_channels=256, out_channels=256, kernel_size=3,
            activation=activation, normalization=normalization)
        self.en_block3_3 = EncoderBlock(
            in_channels=256, out_channels=256, kernel_size=3,
            activation=activation, normalization=normalization)

        # Decoder Blocks: similar to encoder but with single block
        # per level.
        # Upsizing reduced the number of features by 2 (1x1x1 con) and
        # doubled spatial dimension (trilinear interpolation).
        # Skip connection of the same encoded level is added.
        # Final result is obtained by a 1x1x1 conv and a sigmoid function.
        self.de_up2 = LinearUpSampling(
            in_channels=256, out_channels=128, mode=mode)
        self.de_block2 = DecoderBlock(
            in_channels=128, out_channels=128, kernel_size=3,
            activation=activation, normalization=normalization)
        self.de_up1 = LinearUpSampling(
            in_channels=128, out_channels=64, mode=self.mode)
        self.de_block1 = DecoderBlock(
            in_channels=64, out_channels=64, kernel_size=3,
            activation=activation, normalization=normalization)
        self.de_up0 = LinearUpSampling(
            in_channels=64, out_channels=32, mode=mode)
        self.de_block0 = DecoderBlock(
            in_channels=32, out_channels=32, kernel_size=3,
            activation=activation, normalization=normalization)
        self.de_end = OutputTransition(
            in_channels=32, out_channels=num_classes)

        # Variational Auto-Encoder: reduce the input to a low dimensional
        # space of 256 (128 to represent the mean and 128 to represent the std)
        # A sample is drawn from the Gaussian distribution and reconstructed
        # into the inputt image shape following the decoder architecture
        # without interlevel skip connections.
        if self.with_vae:
            self.shapes = self._downsample_shape(
                self.input_shape,
                nb_iterations=4,
                scale_factor=2)
            logger.debug("Shapes: {0}".format(self.shapes))
            self.vae = VAE(
                shapes=self.shapes,
                in_channels=256,
                out_channels=self.in_channels,
                kernel_size=3,
                activation=activation,
                normalization=normalization,
                mode=mode)

    def _downsample_shape(self, shape, nb_iterations=1, scale_factor=2):
        shape = np.asarray(shape)
        all_shapes = [shape.astype(int).tolist()]
        for idx in range(nb_iterations):
            shape = np.ceil(shape / scale_factor)
            all_shapes.append(shape.astype(int).tolist())
        return all_shapes

    def forward(self, x):
        logger.debug("NVnet...")
        logger.debug("Tensor: {0}".format(x.shape))
        out_init = self.in_conv0(x)
        logger.debug("Initial conv: {0}".format(out_init.shape))
        out_en0 = self.en_block0(out_init)
        out_en1 = self.en_block1_1(self.en_block1_0(self.en_down1(out_en0)))
        logger.debug("Encoding block 1: {0}".format(out_en1.shape))
        out_en2 = self.en_block2_1(self.en_block2_0(self.en_down2(out_en1)))
        logger.debug("Encoding block 2: {0}".format(out_en2.shape))
        out_en3 = self.en_block3_3(self.en_block3_2(self.en_block3_1(
            self.en_block3_0(self.en_down3(out_en2)))))
        logger.debug("Encoding block 3: {0}".format(out_en3.shape))
        out_de2 = self.de_block2(self.de_up2(out_en3, out_en2))
        logger.debug("Decoding block 1: {0}".format(out_de2.shape))
        out_de1 = self.de_block1(self.de_up1(out_de2, out_en1))
        logger.debug("Decoding block 2: {0}".format(out_de1.shape))
        out_de0 = self.de_block0(self.de_up0(out_de1, out_en0))
        logger.debug("Decoding block 3: {0}".format(out_de0.shape))
        out_end = self.de_end(out_de0)
        logger.debug("Final conv: {0}".format(out_end.shape))
        if self.with_vae:
            out_vae, out_distr = self.vae(out_en3)
            logger.debug("VAE: {0} - {1}".format(
                out_vae.shape, out_distr.shape))
            out_final = torch.cat((out_end, out_vae), 1)
            return [out_final, out_distr]
        else:
            return out_end


class DownSampling(nn.Module):
    """ A convolution and a padding.
    """
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3,
                 padding=1, dropout_rate=None, bias=False):
        super(DownSampling, self).__init__()
        self.dropout_flag = False
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        if dropout_rate is not None:
            self.dropout_flag = True
            self.dropout = nn.Dropout3d(dropout_rate, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        if self.dropout_flag:
            out = self.dropout(out)
        return out


class EncoderBlock(nn.Module):
    """ Encoder block
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3,
                 padding=1, num_groups=8, activation="relu",
                 normalization="group_normalization"):
        super(EncoderBlock, self).__init__()
        if normalization == "group_normalization":
            self.norm1 = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=in_channels)
            self.norm2 = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=in_channels)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.actv2(out)
        out = self.conv2(out)
        out += residual
        return out


class LinearUpSampling(nn.Module):
    """ Interpolate to upsample.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2,
                 mode="trilinear", align_corners=True):
        super(LinearUpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x, skipx=None, cat=True):
        out = self.conv1(x)
        if skipx is not None:
            if isinstance(skipx, torch.Tensor):
                shape = skipx.shape[2:]
            else:
                shape = skipx
            out = nn.functional.interpolate(
                out,
                size=shape,
                mode=self.mode,
                align_corners=self.align_corners)
        else:
            out = nn.functional.interpolate(
                out,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners)
        if cat and skipx is not None:
            out = torch.cat((out, skipx), 1)
            out = self.conv2(out)
        return out


class DecoderBlock(EncoderBlock):
    """ Decoder block.
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3,
                 padding=1, num_groups=8, activation="relu",
                 normalization="group_normalization"):
        super(DecoderBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            num_groups=num_groups,
            activation=activation,
            normalization=normalization)


class OutputTransition(nn.Module):
    """ Decoder output layer: output the prediction of the segmentation.
    """
    def __init__(self, in_channels, out_channels):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1(x)


class VDResampling(nn.Module):
    """ Variational Auto-Encoder Resampling block.
    """
    def __init__(self, in_channels=256, out_channels=256,
                 dense_features=(10, 12, 8), stride=2, kernel_size=3,
                 padding=1, activation="relu",
                 normalization="group_normalization"):
        super(VDResampling, self).__init__()
        self.mid_chans = int(in_channels / 2)
        self.dense_features = dense_features
        if normalization == "group_normalization":
            self.gn1 = nn.GroupNorm(
                num_groups=8,
                num_channels=in_channels)
        if activation == "relu":
            self.actv1 = nn.ReLU(inplace=True)
            self.actv2 = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.actv1 = nn.ELU(inplace=True)
            self.actv2 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=16, kernel_size=kernel_size,
            stride=stride, padding=padding)
        self.dense1 = nn.Linear(
            in_features=(
                16 * dense_features[0] * dense_features[1] *
                dense_features[2]),
            out_features=in_channels)
        self.dense2 = nn.Linear(
            in_features=self.mid_chans,
            out_features=(
                self.mid_chans * dense_features[0] * dense_features[1] *
                dense_features[2])
            )
        self.up0 = LinearUpSampling(self.mid_chans, out_channels)

    def forward(self, x):
        logger.debug("Resampling tensor: {0}".format(x.shape))
        n_samples = x.shape[0]
        out = self.gn1(x)
        out = self.actv1(out)
        out = self.conv1(out)
        logger.debug("Resampling VD 1.1: {0}".format(out.shape))
        out = out.view(-1, self.num_flat_features(out))
        logger.debug("Resampling VD 1.2: {0}".format(out.shape))
        out_vd = self.dense1(out)
        logger.debug("Resampling VD 2: {0}".format(out_vd.shape))
        distr = out_vd
        out = VDraw(out_vd)
        logger.debug("Resampling VDraw: {0}".format(out.shape))
        out = self.dense2(out)
        out = self.actv2(out)
        logger.debug("Resampling VU 1.1: {0}".format(out.shape))
        out = out.view((n_samples, self.mid_chans, self.dense_features[0],
                        self.dense_features[1], self.dense_features[2]))
        logger.debug("Resampling VU 1.2: {0}".format(out.shape))
        out = self.up0(out, x, cat=False)
        logger.debug("Resampling VU 2: {0}".format(out.shape))
        return out, distr

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def VDraw(x):
    """ Generate a Gaussian distribution with the given mean(128-d) and
    std(128-d).
    """
    return torch.distributions.Normal(x[:, :128], x[:, 128:]).sample()


class VDecoderBlock(nn.Module):
    """ Variational Decoder block.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 activation="relu", normalization="group_normalization",
                 mode="trilinear"):
        super(VDecoderBlock, self).__init__()
        self.up = LinearUpSampling(
            in_channels=in_channels,
            out_channels=out_channels,
            mode=mode)
        self.de_block = DecoderBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            normalization=normalization)

    def forward(self, x, shape=None):
        if shape is None:
            out = self.up(x)
        else:
            out = self.up(x, skipx=shape, cat=False)
        out = self.de_block(out)
        return out


class VAE(nn.Module):
    """ Variational Auto-Encoder: to group the features extracted by Encoder.
    """
    def __init__(self, shapes, in_channels=256, out_channels=4, kernel_size=3,
                 activation="relu", normalization="group_normalization",
                 mode="trilinear"):
        super(VAE, self).__init__()
        self.shapes = shapes
        self.vd_resample = VDResampling(
            in_channels=in_channels,
            out_channels=in_channels,
            dense_features=shapes[-1],
            stride=2,
            kernel_size=kernel_size)
        self.vd_block2 = VDecoderBlock(
            in_channels=in_channels,
            out_channels=in_channels//2,
            kernel_size=kernel_size,
            activation=activation,
            normalization=normalization,
            mode=mode)
        self.vd_block1 = VDecoderBlock(
            in_channels=in_channels//2,
            out_channels=in_channels//4,
            kernel_size=kernel_size,
            activation=activation,
            normalization=normalization,
            mode=mode)
        self.vd_block0 = VDecoderBlock(
            in_channels=in_channels//4,
            out_channels=in_channels//8,
            kernel_size=kernel_size,
            activation=activation,
            normalization=normalization,
            mode=mode)
        self.vd_end = nn.Conv3d(
            in_channels=in_channels//8,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        logger.debug("Variational decoder tensor: {0}".format(x.shape))
        out, distr = self.vd_resample(x)
        logger.debug("Variational decoder resampling: {0} - {1}".format(
            out.shape, distr.shape))
        out = self.vd_block2(out, self.shapes[2])
        logger.debug("Variational decoder 1: {0}".format(out.shape))
        out = self.vd_block1(out, self.shapes[1])
        logger.debug("Variational decoder 2: {0}".format(out.shape))
        out = self.vd_block0(out, self.shapes[0])
        logger.debug("Variational decoder 3: {0}".format(out.shape))
        out = self.vd_end(out)
        logger.debug("Variational decoder final conv: {0}".format(out.shape))
        return out, distr

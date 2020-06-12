# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Unsupervised Learning with CNNs for Image Registration
"""

# Imports
import logging
import collections
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions.normal import Normal
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
from pynet.utils import Regularizers


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family="register")
class VoxelMorphNet(nn.Module):
    """ VoxelMorphNet.

    An unsupervised learning-based inference algorithm that uses insights from
    classical registration methods and makes use of recent developments
    inconvolutional neural networks (CNNs).

    VoxelMorph assumes that input images are pre-affined by an external tool.

    2018 CVPR implementation of voxelmorph.

    TODO: expand this model by including anatomical surface alignment, which
    enables training the network given (optional) anatomical segmentations ->
    described in the paper.

    Reference: https://arxiv.org/abs/1903.03545.
    Code: https://github.com/voxelmorph/voxelmorph.
    """

    def __init__(self, vol_size, enc_nf=[16, 32, 32, 32],
                 dec_nf=[32, 32, 32, 32, 32, 16, 16], full_size=True):
        """ Init class.

        Parameters
        ----------
        vol_size: uplet
            volume size of the atlas.
        enc_nf: list of int, default [16, 32, 32, 32]
            the number of features maps for encoding stages.
        dec_nf: int, default [32, 32, 32, 32, 32, 16, 16]
            the number of features maps for decoding stages.
        full_size: bool, default False
            full amount of decoding layers.
        """
        # Inheritance
        super(VoxelMorphNet, self).__init__()

        # Estimate the generative model mean and covariance using a UNet-style
        # architecture:
        # the network includes a convolutional layer with 32 filters, four
        # downsampling layerswith 64 convolutional filters and a stride of
        # two, and threeupsampling convolutional layers with 64 filters. We
        # onlyupsample three times to predict the velocity field (and
        # following integration steps) at every two voxels, to enablethese
        # operations to fit in current GPU card memory.
        dim = len(vol_size)
        self.unet = UNetCore(dim, enc_nf, dec_nf, full_size)

        # One convolution to get the flow field.
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # Finally warp the moving image.
        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, x):
        """ Forward method.

        Parameters
        ----------
        x: Tensor
            concatenated moving and fixed images.
        """
        logger.debug("VoxelMorphNet...")
        logger.debug("Moving + Fixed: {0}".format(x.shape))
        x = self.unet(x)
        logger.debug("Unet: {0}".format(x.shape))
        flow = self.flow(x)
        logger.debug("Flow: {0}".format(flow.shape))
        moving = x[:, :1]
        logger.debug("Moving: {0}".format(moving.shape))
        warp, _ = self.spatial_transform(moving, flow)
        logger.debug("Warp: {0}".format(warp.shape))
        logger.debug("Done.")
        return warp, {"flow": flow}


class SpatialTransformer(nn.Module):
    """ Represesents a spatial transformation block that uses the output from
    the UNet to preform a grid_sample.
    """
    def __init__(self, size, mode="bilinear"):
        """ Initilaize the block.

        Parameters
        ----------
        size: uplet
            the size of input of the spatial transformer block.
        mode: str, default 'bilinear'
            method of interpolation for the grid sampler.
        """
        # Inheritance
        super(SpatialTransformer, self).__init__()
        self.mode = mode

        # Create sampling grid.
        vectors = [torch.arange(0, val) for val in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

    def forward(self, moving, flow):
        logger.debug("Grid: {0}".format(self.grid.shape))
        new_locs = self.grid + flow
        logger.debug("Field: {0}".format(new_locs.shape))
        shape = flow.shape[2:]
        logger.debug("Shape: {0}".format(shape))

        # Need to normalize grid values to [-1, 1] for resampler
        logger.debug("Normalize field...")
        for idx in range(len(shape)):
            new_locs[:, idx, ...] = (
                2 * (new_locs[:, idx, ...] / (shape[idx] - 1) - 0.5))
        logger.debug("Done...")

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        logger.debug("Field: {0}".format(new_locs.shape))
        warp = func.grid_sample(moving, new_locs, mode=self.mode,
                                align_corners=False)

        return warp, new_locs


class UNetCore(nn.Module):
    """ Class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field.
    """
    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """ Initiliza the UNet model.

        Parameters
        ----------
        enc_nf: list of int, default [16, 32, 32, 32]
            the number of features maps for encoding stages.
        dec_nf: int, default [32, 32, 32, 32, 32, 16, 16]
            the number of features maps for decoding stages.
        full_size: bool, default False
            full amount of decoding layers.
        """
        # Inheritance
        super(UNetCore, self).__init__()
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for idx in range(len(enc_nf)):
            prev_nf = 2 if idx == 0 else enc_nf[idx - 1]
            self.enc.append(ConvBlock(dim, prev_nf, enc_nf[idx], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(ConvBlock(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(ConvBlock(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(ConvBlock(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(ConvBlock(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(ConvBlock(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(ConvBlock(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = ConvBlock(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        """ Forward method.

        Parameters
        ----------
        x: Tensor
            concatenated moving and fixed images.
        """
        logger.debug("UNet...")
        logger.debug("Moving + Fixed: {0}".format(x.shape))

        # Get encoder activations
        x_enc = [x]
        for enc in self.enc:
            logger.debug("Encoder: {0}".format(enc))
            logger.debug("Encoder input: {0}".format(x_enc[-1].shape))
            x_enc.append(enc(x_enc[-1]))
            logger.debug("Encoder output: {0}".format(x_enc[-1].shape))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for idx in range(3):
            logger.debug("Decoder: {0}".format(self.dec[idx]))
            logger.debug("Decoder input: {0}".format(y.shape))
            y = self.dec[idx](y)
            logger.debug("Decoder output: {0}".format(y.shape))
            y = self.upsample(y)
            logger.debug("Decoder upsampling: {0}".format(y.shape))
            y = torch.cat([y, x_enc[-(idx + 2)]], dim=1)
            logger.debug("Decoder skip connexion: {0}".format(y.shape))

        # Two convs at full_size/2 res
        logger.debug("Decoder: {0}".format(self.dec[3]))
        logger.debug("Decoder input: {0}".format(y.shape))
        y = self.dec[3](y)
        logger.debug("Decoder output: {0}".format(y.shape))
        y = self.dec[4](y)
        logger.debug("Decoder: {0}".format(self.dec[4]))
        logger.debug("Decoder input: {0}".format(y.shape))
        logger.debug("Decoder output: {0}".format(y.shape))

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            logger.debug("Full size Decoder upsampling: {0}".format(y.shape))
            y = torch.cat([y, x_enc[0]], dim=1)
            logger.debug("Decoder skip connexion: {0}".format(y.shape))
            logger.debug("Decoder: {0}".format(self.dec[5]))
            logger.debug("Decoder input: {0}".format(y.shape))
            y = self.dec[5](y)
            logger.debug("Decoder output: {0}".format(y.shape))

        # Extra conv for vm2
        if self.vm2:
            logger.debug("VM2: {0}".format(self.vm2_conv))
            logger.debug("VM2 input: {0}".format(y.shape))
            y = self.vm2_conv(y)
            logger.debug("VM2 output: {0}".format(y.shape))

        logger.debug("Done.")

        return y


class ConvBlock(nn.Module):
    """ Represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """
    def __init__(self, dim, in_channels, out_channels, stride=1):
        """ Initialize the conv block.

        Parameters
        ----------
        dim: int
            the number of dimensions of the input.
        in_channels: int
            the number of input channels.
        out_channels: int
            the number of output channels.
        stride: int, default 1
            the stride of the convolution.
        """
        # Inheritance
        super(ConvBlock, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception("Stride must be 1 or 2.")

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


@Regularizers.register
class FlowRegularizer(object):
    """ Total Variation Loss (Smooth Term).

    For a dense flow field, we regularize it with the following loss that
    discourages discontinuity.

    k1 * FlowLoss

    FlowLoss: a gradient loss on the flow field.
    Recommend for k1 are 1.0 for ncc, or 0.01 for mse.
    """
    def __init__(self, k1=0.01):
        self.k1 = k1

    def __call__(self, signal):
        logger.debug("Compute flow regularization...")
        flow = signal.layer_outputs["flow"]
        logger.debug("  lambda: {0}".format(self.k1))
        self.debug("flow", flow)
        flow_loss = self._gradient_loss(flow, penalty="l2")
        logger.debug("  flow loss: {0}".format(flow_loss))
        logger.debug("  flow loss: {0} - {1}".format(flow.min(), flow.max()))
        logger.debug("Done.")
        return self.k1 * flow_loss

    def _gradient_loss(self, flow, penalty="l2"):
        """ Gradient Loss.
        """
        dx = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        if (penalty == "l2"):
            dx = dx * dx
            dy = dy * dy
            dz = dz * dz
        displacement = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return displacement / 3.0

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

# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Volume Tweening Network (VTN) and Affine and Dense Deformable Network (ADDNet)
for Unsupervised medical Image Registration.
"""

# Imports
import logging
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from .voxelmorphnet import SpatialTransformer
from pynet.interfaces import DeepLearningDecorator


# Global parameters
logger = logging.getLogger("pynet")


@DeepLearningDecorator(family="register")
class VTNet(nn.Module):
    """
    VTNet.

    Volume Tweening Network(VTN) consists of several cascaded registration
    subnetworks, after each of which the moving image is warped. The
    unsupervisedtraining of network parameters is guided by the dissimilarity
    between the fixed image and each of the warped images, with the
    regularization losses on the flows predicted by the networks.

    It follows an encoder-decoder architecture.

    Reference: https://arxiv.org/pdf/1902.05020.
    Code: https://github.com/microsoft/Recursive-Cascaded-Networks.
    """
    def __init__(self, input_shape, in_channels, kernel_size=3, padding=1,
                 flow_multiplier=1., nb_channels=16):
        """ Init class.

        Parameters
        ----------
        input_shape: uplet
            the tensor data shape (X, Y, Z).
        in_channels: int
            number of channels in the input tensor.
        kernel_size: int, default 3
            the convolution kernels size (odd number).
        padding: int, default 1
            the padding size, recommended (kernel_size - 1) / 2
        flow_multiplier: foat, default 1
            weight the flow field by this factor.
        nb_channels: int, default 16
            the number of channels after the first convolution.
        """
        # Inheritance
        nn.Module.__init__(self)

        # Class parameters
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.flow_multiplier = flow_multiplier
        self.shapes = self._downsample_shape(
            input_shape, nb_iterations=6, scale_factor=2)
        self.nb_channels = nb_channels

        # Use strided 3D convolution to progressively downsample the image,
        # and then use deconvolution (transposed  convolution) to recover
        # spatial resolution. As suggested in U-Net, skip connections between
        # the  convolutional layers and the deconvolutional layers are added
        # to help refining dense prediction. The network will output the dense
        # flow field, a volume feature map with 3 channels (x,y,z
        # displacements)of the same size as the input.
        out_channels = nb_channels
        for idx in range(1, 3):
            ops = self._conv(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=2, padding=1, bias=True,
                negative_slope=0.1)
            setattr(self, "down{0}".format(idx), ops)
            in_channels = out_channels
            out_channels *= 2
        for idx in range(3, 7):
            ops = self._double_conv(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=2, padding=1, bias=True,
                negative_slope=0.1)
            setattr(self, "down{0}".format(idx), ops)
            in_channels = out_channels
            out_channels *= 2
        out_channels = in_channels // 2
        for idx in range(5, 0, -1):
            pred_ops = self._prediction(in_channels=in_channels)
            ops = self._upconv(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=4, stride=2, groups=1, negative_slope=0.1)
            setattr(self, "pred{0}".format(idx + 1), pred_ops)
            setattr(self, "up{0}".format(idx), ops)
            in_channels = out_channels
            out_channels = out_channels // 2
        self.pred1 = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=3, kernel_size=4,
            stride=2, groups=1)

        # Finally warp the moving image.
        self.spatial_transform = SpatialTransformer(input_shape)

        # Init weights
        @torch.no_grad()
        def weights_init(module):
            if isinstance(module, nn.Conv3d):
                logger.debug("Init weights of {0}...".format(module))
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant(module.bias, 0)
        self.apply(weights_init)

    def _conv(self, in_channels, out_channels, kernel_size, stride=1,
              padding=1, bias=True, negative_slope=1e-2):
        ops = nn.Sequential(collections.OrderedDict([
            ("conv", nn.Conv3d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=bias)),
            ("act", nn.LeakyReLU(negative_slope=negative_slope))
        ]))
        return ops

    def _double_conv(self, in_channels, out_channels, kernel_size, stride=2,
                     padding=1, bias=True, negative_slope=1e-2):
        ops = nn.Sequential(collections.OrderedDict([
            ("conv1", nn.Conv3d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=bias)),
            ("act1", nn.LeakyReLU(negative_slope=negative_slope)),
            ("conv2", nn.Conv3d(out_channels, out_channels, kernel_size,
                                stride=1, padding=padding, bias=bias)),
            ("act2", nn.LeakyReLU(negative_slope=negative_slope))
        ]))
        return ops

    def _upconv(self, in_channels, out_channels, kernel_size, stride=2,
                groups=1, negative_slope=1e-2):
        ops = nn.Sequential(collections.OrderedDict([
            ("convt", nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride=stride,
                groups=groups)),
            ("act", nn.LeakyReLU(negative_slope=negative_slope))
        ]))
        return ops

    def _prediction(self, in_channels):
        ops = nn.Sequential(collections.OrderedDict([
            ("conv", nn.Conv3d(
                in_channels, out_channels=3, kernel_size=3,
                stride=1, padding=1, bias=True)),
            ("convt", nn.ConvTranspose3d(
                in_channels=3, out_channels=3, kernel_size=4, stride=2,
                groups=1))
        ]))
        return ops

    def _downsample_shape(self, shape, nb_iterations=1, scale_factor=2):
        shape = np.asarray(shape)
        all_shapes = [shape.astype(int).tolist()]
        for idx in range(nb_iterations):
            shape = np.ceil(shape / scale_factor)
            all_shapes.append(shape.astype(int).tolist())
        return all_shapes

    def forward(self, x):
        """ Forward method.

        Parameters
        ----------
        x: Tensor
            concatenated moving and fixed images (batch, 2 * channels, X, Y, Z)
        """
        logger.debug("ADDNet...")
        nb_channels = x.shape[1] / 2
        device = x.get_device()

        skipx = []
        for idx in range(1, 7):
            logger.debug("Applying down{0}...".format(idx))
            logger.debug(" input: {0} - {1} - {2}".format(
                x.shape, x.get_device(), x.dtype))
            layer = getattr(self, "down{0}".format(idx))
            logger.debug("  filter: {0}".format(layer))
            x = layer(x)
            skipx.append(x)
            logger.debug("  output: {0} - {1} - {2}".format(
                x.shape, x.get_device(), x.dtype))
            logger.debug("Done.")

        for idx in range(5, 0, -1):
            logger.debug("Applying up{0}...".format(idx))
            logger.debug(" input: {0} - {1} - {2}".format(
                x.shape, x.get_device(), x.dtype))
            layer = getattr(self, "up{0}".format(idx))
            pred_layer = getattr(self, "pred{0}".format(idx + 1))
            logger.debug("  filter: {0}".format(layer))
            logger.debug("  pred filter: {0}".format(pred_layer))
            flow_pred = pred_layer(x)
            x = layer(x)
            x = torch.cat((skipx[idx], x, flow_pred), dim=1)
            logger.debug("  output: {0} - {1} - {2}".format(
                x.shape, x.get_device(), x.dtype))
            logger.debug("Done.")

        logger.debug("Estimating flow field...")
        flow = self.pred1(x)
        logger.debug("Done.")

        logger.debug("Applying warp...")
        moving = x[:, :nb_channels]
        logger.debug("  moving: {0} - {1} - {2}".format(
            moving.shape, moving.get_device(), moving.dtype))
        warp = self.spatial_transform(moving, flow)
        logger.debug("  warp: {0} - {1} - {2}".format(
            warp.shape, warp.get_device(), warp.dtype))
        logger.debug("Done.")

        logger.debug("Done.")

        return warp, {"flow": flow * 20 * self.flow_multiplier}


@DeepLearningDecorator(family="register")
class ADDNet(nn.Module):
    """
    ADDNet.

    Affine and Dense Deformable Network (ADDNet): affine registration
    subnetwork predicts a set of affine parameters, after which a flow field
    can be generated for warping.

    Reference: https://arxiv.org/pdf/1902.05020.
    Code: https://github.com/microsoft/Recursive-Cascaded-Networks.
    """
    def __init__(self, input_shape, in_channels, kernel_size=3, padding=1,
                 flow_multiplier=1.):
        """ Init class.

        Parameters
        ----------
        input_shape: uplet
            the tensor data shape (X, Y, Z).
        in_channels: int
            number of channels in the input tensor.
        kernel_size: int, default 3
            the convolution kernels size (odd number).
        padding: int, default 1
            the padding size, recommended (kernel_size - 1) / 2
        flow_multiplier: foat, default 1
            weight the flow field by this factor.
        """
        # Inheritance
        nn.Module.__init__(self)

        # Class parameters
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.flow_multiplier = flow_multiplier
        self.shapes = self._downsample_shape(
            input_shape, nb_iterations=6, scale_factor=2)
        self.dense_features = np.prod(self.shapes[-1])

        # The input is downsampled by strided 3D convolutions, and finally a
        # fully-connected layer is applied to produce 12 numeric parameters
        # as output, which represents a 3×3 transform matrix.
        out_channels = 16
        for idx in range(1, 3):
            ops = self._conv(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=2, padding=1, bias=True,
                negative_slope=0.1)
            setattr(self, "layer{0}".format(idx), ops)
            in_channels = out_channels
            out_channels *= 2
        for idx in range(3, 7):
            ops = self._double_conv(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=2, padding=1, bias=True,
                negative_slope=0.1)
            setattr(self, "layer{0}".format(idx), ops)
            in_channels = out_channels
            out_channels *= 2
        self.linear1 = torch.nn.Linear(in_channels * self.dense_features, 9)
        self.linear2 = torch.nn.Linear(in_channels * self.dense_features, 3)

        # Finally warp the moving image.
        self.spatial_transform = SpatialTransformer(input_shape)

        # Init weights
        @torch.no_grad()
        def weights_init(module):
            if isinstance(module, nn.Conv3d):
                logger.debug("Init weights of {0}...".format(module))
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant(module.bias, 0)
        self.apply(weights_init)

    def _conv(self, in_channels, out_channels, kernel_size, stride=1,
              padding=1, bias=True, negative_slope=1e-2):
        ops = nn.Sequential(collections.OrderedDict([
            ("conv", nn.Conv3d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=bias)),
            ("act", nn.LeakyReLU(negative_slope=negative_slope))
        ]))
        return ops

    def _double_conv(self, in_channels, out_channels, kernel_size, stride=2,
                     padding=1, bias=True, negative_slope=1e-2):
        ops = nn.Sequential(collections.OrderedDict([
            ("conv1", nn.Conv3d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, bias=bias)),
            ("act1", nn.LeakyReLU(negative_slope=negative_slope)),
            ("conv2", nn.Conv3d(out_channels, out_channels, kernel_size,
                                stride=1, padding=padding, bias=bias)),
            ("act2", nn.LeakyReLU(negative_slope=negative_slope))
        ]))
        return ops

    def _downsample_shape(self, shape, nb_iterations=1, scale_factor=2):
        shape = np.asarray(shape)
        all_shapes = [shape.astype(int).tolist()]
        for idx in range(nb_iterations):
            shape = np.ceil(shape / scale_factor)
            all_shapes.append(shape.astype(int).tolist())
        return all_shapes

    def _affine_flow(self, mat_w, vec_b, shape):
        """ Compute flow field from affine matrix.

        displacement(x) = place(x) - x = (Ax + b) - x = Wx + b
        """
        flow = None
        vec_b = vec_b.view(-1, 1, 1, 1, 3)
        grid = []
        vec_w = []
        for idx, size in enumerate(shape):
            reshape_size = [1] * 5
            reshape_size[idx + 1] = -1
            vec = torch.arange(
                -(size - 1) / 2.0, size / 2.0, 1.0, dtype=torch.FloatTensor)
            grid.append(vec.view(tuple(reshape_size)))
            vec_wi = mat_w[:, :, idx]
            vec_w.append(vec_wi.view(-1, 1, 1, 1, 3))
            if flow is None:
                flow = grid[-1] * vec_w[-1]
            else:
                flow += grid[-1] * vec_w[-1]
        flow += vec_b
        return flow

    def forward(self, x):
        """ Forward method.

        y = Ax + b
        the model learns W = A - I

        Parameters
        ----------
        x: Tensor
            concatenated moving and fixed images (batch, 2 * channels, X, Y, Z)
        """
        logger.debug("ADDNet...")
        nb_channels = x.shape[1] / 2
        device = x.get_device()

        for idx in range(1, 7):
            logger.debug("Applying layer{0}...".format(idx))
            logger.debug(" input: {0} - {1} - {2}".format(
                x.shape, x.get_device(), x.dtype))
            layer = getattr(self, "layer{0}".format(idx))
            logger.debug("  filter: {0}".format(layer))
            x = layer(x)
            logger.debug("  output: {0} - {1} - {2}".format(
                x.shape, x.get_device(), x.dtype))
        logger.debug("Flatening...")
        logger.debug("  input: {0} - {1} - {2}".format(
            x.shape, x.get_device(), x.dtype))
        logger.debug(" dense features: {0}".format(self.dense_features))
        x = x.view(-1, 512 * self.dense_features)
        logger.debug("  output: {0} - {1} - {2}".format(
            x.shape, x.get_device(), x.dtype))
        logger.debug("Getting W...")
        vec_w = self.linear1(x)
        logger.debug("  W: {0} - {1} - {2}".format(
            vec_w.shape, vec_w.get_device(), vec_w.dtype))
        logger.debug("Getting b...")
        vec_b = self.linear2(x)
        logger.debug("  b: {0} - {1} - {2}".format(
            vec_b.shape, vec_b.get_device(), vec_b.dtype))

        logger.debug("Getting A...")
        mat_id = torch.eye(3).to(device)
        mat_id = mat_id.view(1, 3, 3)
        mat_w = vec_w.view(-1, 3, 3) * self.flow_multiplier
        vec_b = vec_b * self.flow_multiplier
        mat_a = mat_w + mat_id

        logger.debug("Getting flow...")
        flow = self._affine_flow(mat_w, vec_b, self.input_shape)

        logger.debug("Applying warp...")
        moving = x[:, :nb_channels]
        logger.debug("  moving: {0} - {1} - {2}".format(
            moving.shape, moving.get_device(), moving.dtype))
        warp = self.spatial_transform(moving, flow)
        logger.debug("  warp: {0} - {1} - {2}".format(
            warp.shape, warp.get_device(), warp.dtype))

        logger.debug("Done.")

        return warp, {"flow": flow, "A": mat_a, "b": vec_b, "W": mat_w}


class ADDNetRegularizer(object):
    """ ADDNet Combined Regularization.

    In addition to the correlation coefficient as our similarity loss, the
    orthogonality loss and the determinant loss are used as regularization
    losses for the affine network.

    k1 * DetLoss + k2 * OrthoLoss

    DetLoss: determinant should be close to 1, ie. reflection is not allowed.
    OrthoLoss: should be close to being orthogonal, ie. penalize the network
    for producing overly non-rigid transform.
    Let C=A'A, a positive semi-definite matrix should be close to I.
    For this, we require C has eigen values close to 1 by minimizing
    k1 + 1/k1 + k2 + 1/k2 + k3 + 1/k3. To prevent NaN, minimize
    k1+eps + (1+eps)^2/(k1+eps) + ...
    """
    def __init__(self, k1=0.1, k2=0.1, eps=1e-5):
        self.k1 = k1
        self.k2 = k2
        self.eps = eps
        self.det_loss = 0
        self.ortho_loss = 0

    def __call__(self, signal):
        mat_a = signal.layer_outputs["A"]
        device = mat_a.get_device()
        det = mat_a.det()
        self.det_loss = torch.norm(det - 1., 2)

        mat_eps = torch.eye(3).to(device) * self.eps
        mat_eps = mat_eps.view(1, 3, 3)
        mat_c = torch.matmul(torch.t(mat_a), mat_a) + mat_eps

        def elem_sym_polys_of_eigen_values(mat):
            mat = [[mat[:, idx_i, idx_j]
                   for idx_j in range(3)] for idx_i in range(3)]
            sigma1 = torch.tensor([mat[0][0], mat[1][1], mat[2][2]])
            sigma2 = torch.tensor([
                mat[0][0] * mat[1][1],
                mat[1][1] * mat[2][2],
                mat[2][2] * mat[0][0]
            ]) - torch.tensor([
                mat[0][1] * mat[1][0],
                mat[1][2] * mat[2][1],
                mat[2][0] * mat[0][2]
            ])
            sigma3 = torch.tensor([
                mat[0][0] * mat[1][1] * mat[2][2],
                mat[0][1] * mat[1][2] * mat[2][0],
                mat[0][2] * mat[1][0] * mat[2][1]
            ]) - torch.tensor([
                mat[0][0] * mat[1][2] * mat[2][1],
                mat[0][1] * mat[1][0] * mat[2][2],
                mat[0][2] * mat[1][1] * mat[2][0]
            ])
            return sigma1, sigma2, sigma3

        s1, s2, s3 = elem_sym_polys_of_eigen_values(mat_c)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        self.ortho_loss = self.k2 * torch.sum(ortho_loss)

        return self.k1 * self.det_loss + self.k2 * self.ortho_loss

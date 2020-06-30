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
from pynet.utils import Networks
from pynet.utils import Regularizers


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family="register")
class VTNet(nn.Module):
    """
    VTNet.

    Volume Tweening Network(VTN) consists of several cascaded registration
    subnetworks, after each of which the moving image is warped. The
    unsupervised training of network parameters is guided by the dissimilarity
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
        # flow field, a volume feature map with 3 channels (X, Y, Z
        # displacements) of the same size as the input.
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
            if idx < 5:
                in_channels = in_channels * 2 + 3
            pred_ops = self._prediction(in_channels=in_channels)
            ops = self._upconv(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=4, stride=2, padding=2, groups=1,
                negative_slope=0.1)
            setattr(self, "pred{0}".format(idx + 1), pred_ops)
            setattr(self, "up{0}".format(idx), ops)
            in_channels = out_channels
            out_channels = out_channels // 2
        in_channels = in_channels * 2 + 3
        self.pred1 = nn.ConvTranspose3d(
            in_channels=in_channels, out_channels=3, kernel_size=4,
            stride=2, padding=1, groups=1)

        # Finally warp the moving image.
        self.spatial_transform = SpatialTransformer(input_shape)

        # Init weights
        @torch.no_grad()
        def weights_init(module):
            if isinstance(module, nn.Conv3d):
                logger.debug("Init weights of {0}...".format(module))
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)
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
                padding=1, groups=1, negative_slope=1e-2):
        ops = nn.Sequential(collections.OrderedDict([
            ("convt", nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size, stride=stride,
                padding=1, groups=groups)),
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
                padding=1, groups=1))
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
        logger.debug("VTNet...")
        nb_channels = x.shape[1] // 2
        device = x.get_device()
        logger.debug("  nb_channels: {0}".format(nb_channels))
        self.debug("input", x)
        moving = x[:, :nb_channels]
        self.debug("moving", moving)

        skipx = []
        for idx in range(1, 7):
            logger.debug("Applying down{0}...".format(idx))
            self.debug("input", x)
            layer = getattr(self, "down{0}".format(idx))
            logger.debug("  filter: {0}".format(layer))
            x = layer(x)
            skipx.append(x)
            self.debug("output", x)
            logger.debug("Done.")

        for idx in range(5, 0, -1):
            logger.debug("Applying up{0}...".format(idx))
            self.debug("input", x)
            layer = getattr(self, "up{0}".format(idx))
            pred_layer = getattr(self, "pred{0}".format(idx + 1))
            logger.debug("  filter: {0}".format(layer))
            logger.debug("  pred filter: {0}".format(pred_layer))
            flow_pred = pred_layer(x)
            self.debug("flow prediction", flow_pred)
            x = layer(x)
            self.debug("layer output", x)
            self.debug("skip connexion", skipx[idx - 1])
            x = torch.cat((skipx[idx - 1], x, flow_pred), dim=1)
            self.debug("output", x)
            logger.debug("Done.")

        logger.debug("Estimating flow field...")
        logger.debug("  pred filter: {0}".format(self.pred1))
        flow = self.pred1(x)
        self.debug("flow", flow)
        logger.debug("Done.")

        logger.debug("Applying warp...")
        self.debug("moving", moving)
        warp, _ = self.spatial_transform(moving, flow)
        self.debug("warp", warp)
        logger.debug("Done.")

        logger.debug("Done.")

        return warp, {"flow": flow * self.flow_multiplier}

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


@Networks.register
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
        self.linear1 = torch.nn.Linear(in_channels * self.dense_features, 9,
                                       bias=False)
        self.linear2 = torch.nn.Linear(in_channels * self.dense_features, 3,
                                       bias=False)

        # Finally warp the moving image.
        self.spatial_transform = SpatialTransformer(input_shape)

        # Init weights
        @torch.no_grad()
        def weights_init(module):
            if isinstance(module, nn.Conv3d):
                logger.debug("Init weights of {0}...".format(module))
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)
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

    def affine_flow(self, affine, size, without_identity=False):
        """ Generates a flow field given an affine matrix.

        Parameters
        ----------
        affine: Tensor (N, 4, 4)
            an affine transform.
        size tuple
            the target output image size.
        without_identity: bool, defaul False
            set to true if the identity matrix has already been substrated to
            the affine matrix.

        Returns
        -------
        flow: Tensor
            the generated affine flow field.
        """
        if not isinstance(size, list):
            size = list(size)
        device = affine.device
        if not without_identity:
            mat_id = torch.eye(4)
            mat_id = mat_id.view(1, 4, 4)
            if device != -1:
                mat_id = mat_id.to(device)
            affine = affine - mat_id
        n_batch = affine.size(0)
        vectors = [torch.arange(0, val) for val in size]
        for indx in range(len(size)):
            vectors[indx] = vectors[indx] - (size[indx] - 1) // 2
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.type(torch.FloatTensor)
        ones = torch.ones([1] + size, dtype=grid.dtype)
        homography_grid = torch.cat([grid, ones], dim=0)
        homography_grid = homography_grid.view(1, 4, 1, -1).permute(3, 0, 1, 2)
        if device != -1:
            homography_grid = homography_grid.to(device)
        self.debug("grid", homography_grid)
        affine = affine.view(1, n_batch, 4, 4)
        self.debug("affine", affine)
        flow = torch.matmul(affine, homography_grid)
        flow = flow.permute(3, 1, 2, 0)
        flow = flow.view([n_batch, 4] + size)
        return flow[:, :3]

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
        nb_channels = x.shape[1] // 2
        device = x.get_device()
        logger.debug("  nb_channels: {0}".format(nb_channels))
        self.debug("input", x)
        moving = x[:, :nb_channels]
        self.debug("moving", moving)

        for idx in range(1, 7):
            logger.debug("Applying layer{0}...".format(idx))
            self.debug("input", x)
            layer = getattr(self, "layer{0}".format(idx))
            logger.debug("  filter: {0}".format(layer))
            x = layer(x)
            self.debug("output", x)
        logger.debug("Flatening...")
        self.debug("input", x)
        logger.debug("  dense features: {0}".format(self.dense_features))
        x = x.view(-1, 512 * self.dense_features)
        self.debug("output", x)
        logger.debug("Getting W...")
        vec_w = self.linear1(x)
        self.debug("W", vec_w)
        logger.debug("Getting b...")
        vec_b = self.linear2(x)
        self.debug("b", vec_b)

        logger.debug("Getting A...")
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x
        # the model learns W = A - I.
        mat_id = torch.eye(3)
        if device != -1:
            mat_id = mat_id.to(device)
        mat_id = mat_id.view(1, 3, 3)
        mat_w = vec_w.view(-1, 3, 3) * self.flow_multiplier
        vec_b = vec_b * self.flow_multiplier
        mat_a = mat_w + mat_id
        self.debug("A", mat_a)

        logger.debug("Getting flow...")
        self.debug("b", vec_b)
        vec_b = vec_b.view(-1, 3, 1)
        # affine = torch.cat((mat_a, vec_b), dim=2)
        # theta = to_homography(affine)
        # norm_theta = normalize_homography(
        #    theta, shape_src=self.input_shape, shape_dst=self.input_shape)
        # theta = norm_theta[:, :3, :]
        # self.debug("theta", theta)
        # size = [mat_a.size(0), 1] + list(self.input_shape)
        # logger.debug("  size: {0}".format(size))
        # flow = func.affine_grid(theta, size, align_corners=False)
        affine = to_homography(torch.cat((mat_w, vec_b), dim=2))
        self.debug("affine", affine)
        flow = self.affine_flow(
            affine, self.input_shape, without_identity=True)
        self.debug("flow", flow)

        logger.debug("Applying warp...")
        self.debug("moving", moving)
        # warp = func.grid_sample(moving, flow, align_corners=False)
        warp, _ = self.spatial_transform(moving, flow)
        self.debug("warp", warp)

        logger.debug("Done.")

        return warp, {"flow": flow, "A": mat_a, "b": vec_b, "W": mat_w}

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


def to_homography(batch_affine):
    """ Convert batch of affine matrices of size (N, 3, 4) to (N, 4, 4).
    """
    affine = func.pad(batch_affine, [0, 0, 0, 1], "constant", value=0.)
    affine[..., -1, -1] += 1.0
    return affine


def normal_transform_pixel(shape):
    """ Compute the normalization matrix from image size in pixels to [-1, 1].
    """
    tr_mat = torch.tensor([[1.0, 0.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0, -1.0],
                           [0.0, 0.0, 1.0, -1.0],
                           [0.0, 0.0, 0.0, 1.0]])
    for idx in range(len(shape)):
        tr_mat[idx, idx] = tr_mat[idx, idx] * 2.0 / (shape[idx] - 1.0)
    tr_mat = tr_mat.unsqueeze(0)
    return tr_mat


def normalize_homography(affine, shape_src, shape_dst):
    """ Normalize a given homography in pixels to [-1, 1].

    Reference: https://discuss.pytorch.org/t/
               affine-transformation-matrix-paramters-conversion/19522/13

    Parameters
    ----------
    affine: torch.Tensor (N, 4, 4)
        homography/ies from source to destiantion to be normalized.
    shape_src: tuple (3,)
        size of the source image.
    shape_dst: tuple (3,)
        size of the destination image.

    Returns
    -------
    affine: torch.Tensor (N, 4, 4)
        the normalized homography/ies.
    """
    if not torch.is_tensor(affine):
        raise TypeError("Input affine type is not a torch.Tensor.")

    if not (len(affine.shape) == 3 or affine.shape[-2:] == (4, 4)):
        raise ValueError("Input affine must be a Nx4x4.")

    # Parameters
    device = affine.device
    dtype = affine.dtype
    # Compute the transformation pixel/norm for src/dst
    src_norm_trf_src_pix = normal_transform_pixel(shape_src).to(device, dtype)
    src_pix_trf_src_norm = torch.inverse(src_norm_trf_src_pix)
    dst_norm_trf_dst_pix = normal_transform_pixel(shape_dst).to(device, dtype)
    # Compute chain transformations
    dst_norm_trans_src_norm = (
        dst_norm_trf_dst_pix @ (affine @ src_pix_trf_src_norm))
    return dst_norm_trans_src_norm


@Regularizers.register
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
    k1 + eps + (1+eps)^2 / (k1+eps) + ...
    """
    def __init__(self, k1=0.1, k2=0.1, eps=1e-5):
        self.k1 = k1
        self.k2 = k2
        self.eps = eps
        self.det_loss = 0
        self.ortho_loss = 0

    def __call__(self, signal):
        logger.debug("ADDNetRegularizer...")

        mat_a = signal.layer_outputs["A"]
        self.debug("A", mat_a)
        device = mat_a.get_device()
        det = mat_a.det()
        self.debug("determinant", det)
        self.det_loss = torch.norm(det - 1., 2)
        logger.debug("  determinant loss: {0}".format(self.det_loss))

        mat_eps = torch.eye(3)
        if device != -1:
            mat_eps = mat_eps.to(device)
        mat_eps *= self.eps
        mat_eps = mat_eps.view(1, 3, 3)
        self.debug("eps", mat_eps)
        mat_c = torch.bmm(mat_a.permute(0, 2, 1), mat_a) + mat_eps
        self.debug("C", mat_c)

        def elem_sym_polys_of_eigen_values(mat):
            mat = [[mat[:, idx_i, idx_j]
                   for idx_j in range(3)] for idx_i in range(3)]
            sigma1 = (mat[0][0] + mat[1][1] + mat[2][2])
            sigma2 = (mat[0][0] * mat[1][1] +
                      mat[1][1] * mat[2][2] +
                      mat[2][2] * mat[0][0]) - (
                      mat[0][1] * mat[1][0] +
                      mat[1][2] * mat[2][1] +
                      mat[2][0] * mat[0][2])
            sigma3 = (mat[0][0] * mat[1][1] * mat[2][2] +
                      mat[0][1] * mat[1][2] * mat[2][0] +
                      mat[0][2] * mat[1][0] * mat[2][1]) - (
                      mat[0][0] * mat[1][2] * mat[2][1] +
                      mat[0][1] * mat[1][0] * mat[2][2] +
                      mat[0][2] * mat[1][1] * mat[2][0])
            return sigma1, sigma2, sigma3

        s1, s2, s3 = elem_sym_polys_of_eigen_values(mat_c)
        self.debug("s1", s1)
        self.debug("s2", s2)
        self.debug("s3", s3)
        eps = self.eps
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        self.debug("orthogonal", ortho_loss)
        self.ortho_loss = self.k2 * torch.sum(ortho_loss)
        logger.debug("  orthogonal loss: {0}".format(self.ortho_loss))

        logger.debug("Done.")

        return self.k1 * self.det_loss + self.k2 * self.ortho_loss

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

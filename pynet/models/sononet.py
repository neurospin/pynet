# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Sononet is a CNN architecture with two components: a feature extractor module
and an adaptation module.
"""

# Imports
import logging
import collections
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as func
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import numpy as np


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("classifier"))
class SonoNet(nn.Module):
    """ SonoNet.

    Feature extraction: the first 17 layers (counting
    max-pooling) of the VGG network is used to extract discriminant features
    (3 layers for the first 3 and 2 layers for the last 2 feature scales).
    Note that the number of filters are doubled after each of the first
    three max-pooling operations.
    Attention maps (adaptation module): the number of channels are first
    reduced to the number of target classes C. Subsequently, the spatial
    information is flattened via channel-wise global average pooling. Finally,
    a soft-max operation is applied to the resulting vector and the entry
    with maximum activation is selected as the prediction.
    As the network is constrained to classify based on the reduced vector,
    the network is forced to extract the most salient features for each class.

    Reference: Attention-Gated Networksfor Improving Ultrasound Scan Plane
    Detection https://arxiv.org/pdf/1804.05338.pdf
    Code: https://github.com/ozan-oktay/Attention-Gated-Networks
    """

    def __init__(self, n_classes, in_channels=1, n_convs=[3, 3, 3, 2, 2],
                 start_filts=64, batchnorm=True, nonlocal_mode="concatenation",
                 aggregation_mode="concat"):
        """ Init class.

        Parameters
        ----------
        n_classes: int
            the number of features in the output segmentation map.
        in_channels: int, default 1
            number of channels in the input tensor.
        n_convs: list of int, default [3, 3, 3, 2, 2]
            the number of convolutions
        start_filts: int, default 64
            number of convolutional filters for the first conv.
        batchnorm: bool, default False
            normalize the inputs of the activation function.
        nonlocal_mode: str, default 'concatenation'
        aggregation_mode: str, default 'concat'
        """
        # Inheritance
        nn.Module.__init__(self)

        # Parameters
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_convs = n_convs
        self.start_filts = start_filts
        self.batchnorm = batchnorm
        self.nonlocal_mode = nonlocal_mode
        self.aggregation_mode = aggregation_mode

        # Feature Extraction
        filters = [start_filts * cnt for cnt in range(1, len(self.n_convs))]
        self.conv1 = Conv2(
            self.in_channels, filters[0], self.batchnorm, n=n_convs[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = Conv2(
            filters[0], filters[1], self.batchnorm, n=n_convs[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = Conv2(
            filters[1], filters[2], self.batchnorm, n=n_convs[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = Conv2(
            filters[2], filters[3], self.batchnorm, n=n_convs[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = Conv2(
            filters[3], filters[3], self.batchnorm, n=n_convs[4])

        # Attention Maps
        self.compatibility_score1 = GridAttentionBlock2D(
            in_channels=filters[2], gating_channels=filters[3],
            inter_channels=filters[3], sub_sample_factor=(1, 1),
            mode=nonlocal_mode, use_W=False, use_phi=True,
            use_theta=True, use_psi=True, nonlinearity1="relu")

        self.compatibility_score2 = GridAttentionBlock2D(
            in_channels=filters[3], gating_channels=filters[3],
            inter_channels=filters[3], sub_sample_factor=(1, 1),
            mode=nonlocal_mode, use_W=False, use_phi=True,
            use_theta=True, use_psi=True, nonlinearity1="relu")

        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == "concat":
            self.classifier = nn.Linear(filters[2] + filters[3] + filters[3],
                                        n_classes)
            self.aggregate = self.aggregation_concat

        else:
            self.classifier1 = nn.Linear(filters[2], n_classes)
            self.classifier2 = nn.Linear(filters[3], n_classes)
            self.classifier3 = nn.Linear(filters[3], n_classes)
            self.classifiers = [self.classifier1, self.classifier2,
                                self.classifier3]

            if aggregation_mode == "mean":
                self.aggregate = self.aggregation_sep

            elif aggregation_mode == "deep_sup":
                self.classifier = nn.Linear(
                    filters[2] + filters[3] + filters[3], n_classes)
                self.aggregate = self.aggregation_ds

            elif aggregation_mode == "ft":
                self.classifier = nn.Linear(n_classes * 3, n_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

        # Initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def aggregation_sep(self, *attended_maps):
        return [clf(att) for clf, att in zip(self.classifiers, attended_maps)]

    def aggregation_ft(self, *attended_maps):
        preds = self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep = self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))

    def forward(self, inputs):
        logger.debug("SONO Net...")

        logger.debug("Feature Extraction:")
        self.debug("input", inputs)
        conv1 = self.conv1(inputs)
        self.debug("conv1", conv1)
        maxpool1 = self.maxpool1(conv1)
        self.debug("maxpool1", maxpool1)
        conv2 = self.conv2(maxpool1)
        self.debug("conv2", conv2)
        maxpool2 = self.maxpool2(conv2)
        self.debug("maxpool2", maxpool2)
        conv3 = self.conv3(maxpool2)
        self.debug("conv3", conv3)
        maxpool3 = self.maxpool3(conv3)
        self.debug("maxpool3", maxpool3)
        conv4 = self.conv4(maxpool3)
        self.debug("conv4", conv4)
        maxpool4 = self.maxpool4(conv4)
        self.debug("maxpool4", maxpool4)
        conv5 = self.conv5(maxpool4)
        self.debug("conv5", conv5)

        batch_size = inputs.shape[0]
        pooled = func.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)
        self.debug("pooled", pooled)

        logger.debug("Attention Mechanism:")
        g_conv1, att1 = self.compatibility_score1(conv3, conv5)
        self.debug("g_conv1", g_conv1)
        self.debug("att1", att1)
        g_conv2, att2 = self.compatibility_score2(conv4, conv5)
        self.debug("g_conv2", g_conv2)
        self.debug("att2", att2)

        logger.debug("Flatten to get single feature vector:")
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        self.debug("g1", g1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)
        self.debug("g2", g2)

        logger.debug("Aggregate:")
        out = self.aggregate(g1, g2, pooled)
        if self.aggregation_mode == "mean":
            out = [item.view(-1, self.n_classes, 1) for item in out]
            out = torch.cat(out, dim=2)
        self.debug("out", out)

        return out

    @staticmethod
    def apply_softmax(pred):
        log_p = func.softmax(pred, dim=1)
        return log_p

    @staticmethod
    def aggregate_output(output, aggregation="mean",
                         aggregation_weight=[1, 1, 1], idx=0):
        """ Given a list of predictions from the model, make a decision based
        on aggreagation rules.
        """
        if output.ndim == 3:
            logits = []
            for idx in range(output.shape[2]):
                logits.append(SonoNet.apply_softmax(
                    output[:, :, idx]).unsqueeze(dim=0))
            logits = torch.cat(logits, dim=0)
            if aggregation == "max":
                _, pred = logits.data.max(dim=0)[0].max(dim=1)
            elif aggregation == "mean":
                _, pred = logits.mean(dim=0).max(dim=1)
            elif aggregation == "weighted_mean":
                weight_t = torch.from_numpy(np.array(weight, dtype=np.float32))
                aggregation_weight = weight_t.view(-1, 1, 1).to(output.device)
                wlogits = (aggregation_weight.expand_as(logits) * logits)
                _, pred = wlogits.data.mean(dim=0).max(dim=1)
            else:
                _, pred = logits[:, :, idx].data.max(dim=1)
        else:
            logits = SonoNet.apply_softmax(output)
            _, pred = logits.data.max(dim=1)
        return pred

    def debug(self, name, tensor):
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 dimension=3, mode="concatenation",
                 sub_sample_factor=(1, 1, 1), bn_layer=True, use_W=True,
                 use_phi=True, use_theta=True, use_psi=True,
                 nonlinearity1="relu"):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ["concatenation", "concatenation_softmax",
                        "concatenation_sigmoid", "concatenation_mean",
                        "concatenation_range_normalise",
                        "concatenation_mean_flow"]

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = (
            sub_sample_factor if isinstance(sub_sample_factor, tuple)
            else tuple([sub_sample_factor]) * dimension)
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = "trilinear"
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = "bilinear"
        else:
            raise NotImplemented

        # initialise id functions
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.W = lambda x: x
        self.theta = lambda x: x
        self.psi = lambda x: x
        self.phi = lambda x: x
        self.nl1 = lambda x: x

        if use_W:
            if bn_layer:
                self.W = nn.Sequential(
                    conv_nd(in_channels=self.in_channels,
                            out_channels=self.in_channels, kernel_size=1,
                            stride=1, padding=0),
                    bn(self.in_channels),
                )
            else:
                self.W = conv_nd(in_channels=self.in_channels,
                                 out_channels=self.in_channels, kernel_size=1,
                                 stride=1, padding=0)

        if use_theta:
            self.theta = conv_nd(in_channels=self.in_channels,
                                 out_channels=self.inter_channels,
                                 kernel_size=self.sub_sample_kernel_size,
                                 stride=self.sub_sample_factor, padding=0,
                                 bias=False)

        if use_phi:
            self.phi = conv_nd(in_channels=self.gating_channels,
                               out_channels=self.inter_channels,
                               kernel_size=self.sub_sample_kernel_size,
                               stride=self.sub_sample_factor, padding=0,
                               bias=False)

        if use_psi:
            self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1,
                               kernel_size=1, stride=1, padding=0, bias=True)

        if nonlinearity1:
            if nonlinearity1 == "relu":
                self.nl1 = lambda x: func.relu(x, inplace=True)

        if "concatenation" in mode:
            self.operation_function = self._concatenation
        else:
            raise NotImplementedError("Unknown operation function.")

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type="kaiming")

        if use_psi and self.mode == "concatenation_sigmoid":
            nn.init.constant_(self.psi.bias.data, 3.0)

        if use_psi and self.mode == "concatenation_softmax":
            nn.init.constant_(self.psi.bias.data, 10.0)

        # if use_psi and self.mode == "concatenation_mean":
        #     nn.init.constant(self.psi.bias.data, 3.0)

        # if use_psi and self.mode == "concatenation_range_normalise":
        #     nn.init.constant(self.psi.bias.data, 3.0)

    def forward(self, x, g):
        """
        Parameters
        ----------
        x: (b, c, t, h, w)
        g: (b, g_d)
        """
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # compute compatibility score

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w)
        # phi   => (b, c, t, h, w) -> (b, i_c, t, h, w)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        #  nl(theta.x + phi.g + bias) -> f = (b, i_c, t/s1, h/s2, w/s3)
        phi_g = func.interpolate(self.phi(g), size=theta_x_size[2:],
                                 mode=self.upsample_mode, align_corners=True)

        f = theta_x + phi_g
        f = self.nl1(f)

        psi_f = self.psi(f)

        # normalisation -- scale compatibility score
        #  psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
        if self.mode == "concatenation_softmax":
            sigm_psi_f = func.softmax(psi_f.view(batch_size, 1, -1), dim=2)
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == "concatenation_mean":
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            psi_f_sum = torch.sum(psi_f_flat, dim=2)  # clamp(1e-6)
            psi_f_sum = psi_f_sum[:, :, None].expand_as(psi_f_flat)

            sigm_psi_f = psi_f_flat / psi_f_sum
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == "concatenation_mean_flow":
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            ss = psi_f_flat.shape
            psi_f_min = psi_f_flat.min(dim=2)[0].view(ss[0], ss[1], 1)
            psi_f_flat = psi_f_flat - psi_f_min
            psi_f_sum = torch.sum(psi_f_flat, dim=2).view(
                ss[0], ss[1], 1).expand_as(psi_f_flat)

            sigm_psi_f = psi_f_flat / psi_f_sum
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == "concatenation_range_normalise":
            psi_f_flat = psi_f.view(batch_size, 1, -1)
            ss = psi_f_flat.shape
            psi_f_max = torch.max(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)
            psi_f_min = torch.min(psi_f_flat, dim=2)[0].view(ss[0], ss[1], 1)

            sigm_psi_f = (
                (psi_f_flat - psi_f_min) / (psi_f_max - psi_f_min).expand_as(
                    psi_f_flat))
            sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])
        elif self.mode == "concatenation_sigmoid":
            sigm_psi_f = func.sigmoid(psi_f)
        else:
            raise NotImplementedError

        # sigm_psi_f is attention map! upsample the attentions and multiply
        sigm_psi_f = func.interpolate(sigm_psi_f, size=input_size[2:],
                                      mode=self.upsample_mode,
                                      align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class Conv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1,
                 padding=1):
        # Inheritance
        super(Conv2, self).__init__()

        # Parameters
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding

        # Successive convolutions
        for cnt in range(1, n + 1):
            if is_batchnorm:
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, stride, padding),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True))
            else:
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, stride, padding),
                    nn.ReLU(inplace=True))
            setattr(self, "conv{0}".format(cnt), conv)
            in_size = out_size

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        x = inputs
        for cnt in range(1, self.n + 1):
            conv = getattr(self, "conv{0}".format(cnt))
            x = conv(x)
        return x


class Conv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 1),
                 padding_size=(1, 1, 0), init_stride=(1, 1, 1)):
        # Inheritance
        super(Conv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size, init_stride,
                          padding_size),
                nn.BatchNorm3d(out_size),
                nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                nn.BatchNorm3d(out_size),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size, init_stride,
                          padding_size),
                nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                nn.ReLU(inplace=True),)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 mode="concatenation", sub_sample_factor=(1, 1), bn_layer=True,
                 use_W=True, use_phi=True, use_theta=True, use_psi=True,
                 nonlinearity1="relu"):
        super(GridAttentionBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            gating_channels=gating_channels,
            dimension=2, mode=mode,
            sub_sample_factor=sub_sample_factor,
            bn_layer=bn_layer,
            use_W=use_W,
            use_phi=use_phi,
            use_theta=use_theta,
            use_psi=use_psi,
            nonlinearity1=nonlinearity1)


class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None,
                 mode="concatenation", sub_sample_factor=(1, 1, 1),
                 bn_layer=True):
        super(GridAttentionBlock3D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            gating_channels=gating_channels,
            dimension=3, mode=mode,
            sub_sample_factor=sub_sample_factor,
            bn_layer=bn_layer)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "Initialization method {0} is not implemented.".format(init_type))

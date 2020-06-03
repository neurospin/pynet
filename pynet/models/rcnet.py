# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Recursive Cascaded Networks (RCNet) for Unsupervised Medical Image
Registration using and Dense Deformable Network (ADDNet) and Volume Tweening
Network (VTN).
"""

# Imports
import logging
import collections
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from pynet.observable import SignalObject
from pynet.losses import PCCLoss
from .vtnet import ADDNetRegularizer
from .voxelmorphnet import FlowRegularizer
from .voxelmorphnet import SpatialTransformer
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
from pynet.utils import Regularizers
from pynet.utils import get_tools


# Global parameters
Stem = namedtuple("Stem", ["network", "params"])
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family="register")
class RCNet(nn.Module):
    """ RCnet.

    The recursive cascaded networks is a general architecture that enables
    learning deep cascades and can be used for deformable image registration.
    The cascade architecture is simple in design and can be built on any base
    network. The moving image is warped successively by each cascade and
    finally aligned to the fixed image; this procedure is recursive in a way
    that every cascade learns to perform a progressive deformation for the
    current warped image. The entire system is end-to-end and jointly trained
    in an unsupervised manner.
    Shared-weight techniques are developed in addition to the recursive
    architecture. Shared-weight cascading in training is not used since it
    consumes extra GPU memory.

    We use the Dense Deformable Network (ADDNet) to estimate the affine
    transform in combination with a deformation field network estimator.

    This network achieves state-of-the-art performance on both liver CT and
    brain MRI datasets for 3D medical image registration.

    Reference:
    * https://arxiv.org/pdf/1907.12353
    * https://arxiv.org/pdf/1902.05020

    Code:
    * https://github.com/microsoft/Recursive-Cascaded-Networks.
    """
    default_params = {
        "weight": 1.,
        "raw_weight": 1.,
        "reg_weight": 1.}

    def __init__(self, input_shape, in_channels, base_network, n_cascades=1,
                 rep=1):
        """ Init class.

        Parameters
        ----------
        input_shape: uplet
            the tensor data shape (X, Y, Z).
        in_channels: int
            number of channels in the input tensor.
        base_network: str
            the name of the Network used to estimate the non-linear
            deformation.
        n_cascades: int, default 1
            the number of cascades.
        rep: int, default 1
            the number of times of shared-weight cascading.
        """
        # Inheritance
        logger.debug("RCNet configuration...")
        nn.Module.__init__(self)

        # Class parameters
        available_networks = get_tools()["networks"]
        if base_network not in available_networks:
            raise ValueError(
                "Unknown base network '{0}', available networks are "
                "{1}.".format(base_network, available_networks.keys()))
        self.base_network = available_networks[base_network]
        logger.debug("  base network: {0}".format(self.base_network))
        self.stems = [Stem(
            network=available_networks["ADDNet"](
                input_shape=input_shape, in_channels=in_channels,
                flow_multiplier=1.),
            params={"raw_weight": 0, "reg_weight": 0})]
        self.stems += [Stem(
            network=self.base_network(
                input_shape=input_shape, in_channels=in_channels,
                flow_multiplier=(1. / n_cascades)),
            params={"raw_weight": 0})] * (rep * n_cascades)
        self.stems[-1].params["raw_weight"] = 1
        for stem in self.stems:
            for key, val in self.default_params.items():
                if key not in stem.params:
                    stem.params[key] = val
        logger.debug("  stems: {0}".format(self.stems))

        # Finally warp the moving image: avoid accumulation of interpolation
        # errors, ie. reinterpolate after each cascade.
        self.spatial_transform = SpatialTransformer(input_shape)

    def parameters(self):
        """ Get the trainable variables.
        """
        return list(set(
            sum([list(stem.network.parameters()) for stem in self.stems], [])))

    @property
    def trainable_parameters(self):
        """ Get the number of trainable parameters.
        """
        nb_params = 0
        for stem in self.stems:
            nb_params += sum(
                params.numel() for params in stem.network.parameters())
        return nb_params

    def forward(self, x):
        """ Forward method.

        Parameters
        ----------
        x: Tensor
            concatenated moving and fixed images (batch, 2 * channels, X, Y, Z)
        """
        logger.debug("RCNET...")
        device = x.device
        for stem in self.stems:
            if next(stem.network.parameters()).device != device:
                stem.network.to(device)
        stem_results = []
        nb_channels = x.shape[1] // 2
        moving = x[:, :nb_channels]
        fixed = x[:, nb_channels:]
        warp, stem_result = self.stems[0].network(
            torch.cat((moving, fixed), dim=1))
        stem_result["warped"] = warp
        stem_result["agg_flow"] = stem_result["flow"]
        stem_result["stem_params"] = self.stems[0].params
        stem_results.append(stem_result)
        for stem in self.stems[1:]:
            warp, stem_result = stem.network(
                torch.cat((stem_results[-1]["warped"], fixed), dim=1))
            stem_result["stem_params"] = stem.params
            stem_result["agg_flow"] = (
                stem_results[-1]["agg_flow"] + stem_result["flow"])
            stem_result["warped"], _ = self.spatial_transform(
                moving, stem_result["agg_flow"])
            stem_results.append(stem_result)

        flow = stem_results[-1]["agg_flow"]
        warp = stem_results[-1]["warped"]
        jacobian_det = 0  # self.jacobian_det(flow)

        return warp, {"flow": flow, "stem_results": stem_results,
                      "jacobian_det": jacobian_det}

    def jacobian_det(self, flow):
        """ Compute the Jacobian determinant of displacement field.
        """
        # Compute Jacobian row by row.
        jac = [
            flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1] +
            Variable(torch.tensor(
                [1, 0, 0], dtype=torch.float32), requires_grad=False),
            flow[:, :, :-1, 1:, :-1] - flow[:, :, :-1, :-1, :-1] +
            Variable(torch.tensor(
                [0, 1, 0], dtype=torch.float32), requires_grad=False),
            flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1] +
            Variable(torch.tensor(
                [0, 0, 1], dtype=torch.float32), requires_grad=False)
        ]
        jac = torch.stack(jac, dim=1)

        # Take the determinant of the Jacobian
        var = torch.std(torch.det(jac), dim=(2, 3, 4))

        return torch.sqrt(var)


@Regularizers.register
class RCNetRegularizer(object):
    """ RCNet Regularization.

    ADDNetRegularizer + FlowRegularizer.
    """
    def __init__(self, det_factor=0.1, ortho_factor=0.1, reg_factor=1.0):
        self.addnet_reg = ADDNetRegularizer(k1=det_factor, k2=ortho_factor)
        self.flow_reg = FlowRegularizer(k1=reg_factor)

    def __call__(self, signal):
        logger.debug("Compute RCNet regularization...")
        stem_results = signal.layer_outputs["stem_results"]
        for stem_result in stem_results:
            params = stem_results["stem_params"]
            sub_signal = SignalObject()
            setattr(sub_signal, "layer_outputs", stem_result)
            if "W" in stem_result:
                stem_result["loss"] = self.addnet_reg(signal)
            else:
                if params["reg_weight"] > 0:
                    flow_loss = self.flow_reg(signal)
                    stem_result["loss"] = flow_loss * params["reg_weight"]
        loss = sum([
            stem_result["loss"] * stem_results["stem_params"]["weight"]
            for stem_result in stem_results])
        logger.debug("Done.")
        return loss

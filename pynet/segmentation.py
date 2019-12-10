# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define encoder models.
"""

# Third party import
import torch
import torch.nn.functional as func
import numpy as np

# Package import
import pynet.models as models
from pynet.core import Base


class NvNetSegmenter(Base):
    """ NvNet (2018) by Andriy Myronenko.
    """
    def __init__(self, input_shape, num_classes, activation="relu",
                 normalization="group_normalization", mode="trilinear",
                 with_vae=True, pretrained=None, optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 use_cuda=False, **kwargs):
        """ Class initilization.

        Parameters
        ----------
        input_shape: uplet
            the tensor shape (nb_samples, nb_channels, X, Y, Z).
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
        pretrained: path, default None
            path to the pretrained model or weights.
        optimizer_name: str, default 'Adam'
            the name of the optimizer: see 'torch.optim' for a description
            of available optimizer.
        learning_rate: float, default 1e-3
            the optimizer learning rate.
        loss_name: str, default 'NLLLoss'
            the name of the loss: see 'torch.nn' for a description
            of available loss.
        metrics: list of str
            a list of extra metrics that will be computed.
        use_cuda: bool, default False
            wether to use GPU or CPU.
        kwargs: dict
            specify directly a custom 'optimizer' or 'loss'. Can also be used
            to set specific optimizer parameters.
        """
        self.model = models.NvNet(
            input_shape=input_shape,
            num_classes=num_classes,
            activation=activation,
            normalization=normalization,
            mode=mode,
            with_vae=with_vae)
        super().__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)

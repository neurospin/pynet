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


class UNetEncoder(Base):
    """ UNet (2014) by Long and Shelhamer.
    """
    def __init__(self, num_classes, in_channels=1, depth=5, start_filts=64,
                 up_mode="transpose", merge_mode="concat", batchnorm=False,
                 dim="3d", pretrained=None, optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 use_cuda=False, **kwargs):
        """ Class initilization.

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
        dim: str, default '3d'
            '3d' or '2d' input data.
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
        self.model = models.UNet(
            num_classes=num_classes,
            in_channels=in_channels,
            depth=depth,
            start_filts=start_filts,
            up_mode=up_mode,
            merge_mode=merge_mode,
            batchnorm=batchnorm,
            dim=dim)
        super().__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)

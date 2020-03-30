# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that privides common networks.
"""


class RegisteryDecorator(object):
    """ Class that can be used to register class in a registry.
    """
    @classmethod
    def register(cls, klass, *args, **kwargs):
        name = klass.__name__
        if name in cls.REGISTRY:
            raise ValueError(
                "'{0}' name already used in registry.".format(name))
        cls.REGISTRY[name] = klass
        return klass

    @classmethod
    def get_registry(cls):
        return cls.REGISTRY


class Networks(RegisteryDecorator):
    """ Class that register all the available networks.
    """
    REGISTRY = {}


class Regularizers(RegisteryDecorator):
    """ Class that register all the available regularizers.
    """
    REGISTRY = {}


from .unet import UNet
from .nvnet import NvNet
from .voxelmorphnet import VoxelMorphNet
from .vtnet import VTNet, ADDNet
# from .rcnet import RCNet
from .torchvisnet import *

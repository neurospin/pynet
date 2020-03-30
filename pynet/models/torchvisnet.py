# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Import classifier models defined in torchvision.
"""

# Imports
import re
import sys
import torch.nn as nn
from torchvision import models
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks


@Networks.register
@DeepLearningDecorator(family="classifier")
class VGG(models.VGG):
    """ VGGNet (2014) by Simonyan and Zisserman.
    """
    def __init__(self, cfg, num_classes, batch_norm=False, init_weights=True,
                 pretrained=None, make_layers=models.vgg.make_layers):
        """ Class initilization.

        Parameters
        ----------
        cfg: list
            the model features: output channels number for convolution or
            'M' for max pooling.
        num_classes: int
            the number of classes to predict.
        batch_norm: bool, default False
            use batch normalization after each convolution.
        init_weights: bool, default True
            initialize the model weights.
        pretrained: str, default None
            update the weights of the model using this state information.
        make_layers: @func
            a function to create the feature layers: default 2d max pooling
            with kernel size 2 and stride 2, and convolution with kernel size
            3 and padding 1.
        """
        models.VGG.__init__(
            self,
            features=make_layers(cfg, batch_norm=batch_norm),
            num_classes=num_classes,
            init_weights=init_weights)


@Networks.register
@DeepLearningDecorator(family="classifier")
class DenseNet(models.DenseNet):
    """ DenseNet.
    """
    def __init__(self, growth_rate, block_config, num_init_features,
                 num_classes, bn_size=4, drop_rate=0, memory_efficient=False):
        """ Class initilization.

        Parameters
        ----------
        growth_rate: int
            how many filters to add each layer ('k' in paper).
        block_config: 1-uplet
            how many layers in each pooling block.
        num_init_features: int
            the number of filters to learn in the first convolution layer.
        num_classes: int
            number of classification classes.
        bn_size: int, default 4
            multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer).
        drop_rate: float, default 0
            dropout rate after each dense layer.
        memory_efficient: bool, default False
            if True, uses checkpointing. Much more memory efficient,
            but slower.
        """
        models.DenseNet.__init__(
            self,
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            num_classes=num_classes,
            memory_efficient=memory_efficient)


@Networks.register
@DeepLearningDecorator(family="classifier")
class ResNet(models.ResNet):
    """ Residual Neural Network (ResNet) by Kaiming He et al.
    """
    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        """ Class initilization.

        Parameters
        ----------
        block: nn Module
            one block architecture.
        layers: 4-uplet
            control the number of element in each layer.
        num_classes: int
            number of classification classes.
        zero_init_residual: bool, default False
            zero-initialize the last BN in each residual branch, so that the
            residual branch starts with zeros, and each residual block behaves
            like an identity.
        groups: int, default 1
            controls the connections between inputs and outputs during
            convolution.
        width_per_group: int, default 64
            control the number of input and output channels during convolution.
        replace_stride_with_dilation: uplet, default None
            each element in the tuple indicates if we should replace
            the 2x2 stride with a dilated convolution instead.
        norm_layer: nn Module, default None
            use the specified normalization module, by default use batch
            normalization.
        """
        models.ResNet.__init__(
            self,
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer)


@Networks.register
@DeepLearningDecorator(family="classifier")
class Inception3(models.Inception3):
    """ Inception v3 by Google.
    """
    def __init__(self, num_classes, aux_logits=True, transform_input=False):
        """ Class initilization.

        Parameters
        ----------
        num_classes: int
            number of classification classes.
        aux_logits: bool, default False
            auxiliary classifier for the training.
        transform_input: bool, default False
            normalize the data.
        """
        models.Inception3.__init__(
            self,
            num_classes=num_classes,
            aux_logits=aux_logits,
            transform_input=transform_input)


def class_factory(klass_name, klass_params, destination_module_globals):
    """ Dynamically overload a class.

    In order to make the class publicly accessible, we assign the result of
    the function to a variable dynamically using globals().

    Parameters
    ----------
    klass_name: str
        the class name that will be created.
    klass_params: dict
        the class specific parameters.
    """
    class VGGBase(VGG):
        """ VGGNet X-layer.
        """
        cfg = None

        def __init__(self, num_classes, batch_norm=False, init_weights=True):
            if self.cfg is None:
                raise ValueError("Please specify a configuration first.")
            VGG.__init__(
                self,
                cfg=self.cfg,
                num_classes=num_classes,
                batch_norm=batch_norm,
                init_weights=init_weights)

    class DenseNetBase(DenseNet):
        """ DenseNet-X model.
        """
        growth_rate = None
        block_config = None
        num_init_features = None

        def __init__(self, num_classes, bn_size=4, drop_rate=0,
                     memory_efficient=False):
            for name in ("growth_rate", "block_config", "num_init_features"):
                if getattr(self, name) is None:
                    raise ValueError(
                        "Please specify '{0}' first.".format(name))
            DenseNet.__init__(
                self,
                growth_rate=self.growth_rate,
                block_config=self.block_config,
                num_init_features=self.num_init_features,
                num_classes=num_classes,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient)

    class ResNetBase(ResNet):
        """ ResNet-X model.
        """
        block = None
        layers = None
        groups = 1
        width_per_group = 64

        def __init__(self, num_classes, zero_init_residual=False, groups=1,
                     width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None):
            for name in ("block", "layers"):
                if getattr(self, name) is None:
                    raise ValueError(
                        "Please specify '{0}' first.".format(name))
            ResNet.__init__(
                self,
                block=self.block,
                layers=self.layers,
                num_classes=num_classes,
                zero_init_residual=zero_init_residual,
                groups=groups,
                width_per_group=width_per_group,
                replace_stride_with_dilation=replace_stride_with_dilation,
                norm_layer=norm_layer)
    for klass, ref_klass in ((VGGBase, VGG), (DenseNetBase, DenseNet),
                             (ResNetBase, ResNet)):
        klass.__doc__ = ref_klass.__doc__
        klass.__init__.__doc__ = ref_klass.__init__.__doc__
    klass_map = {
        "VGG": VGGBase,
        "DenseNet": DenseNetBase,
        "ResNet": ResNetBase
    }
    klass_params.update({
        "__module__": destination_module_globals["__name__"],
        "_id":  destination_module_globals["__name__"] + "." + klass_name
    })
    klass_base_name = re.findall(r"([a-zA-Z]+)[0-9]+", klass_name)[0]
    decorator = DeepLearningDecorator(family="classifier")
    destination_module_globals[klass_name] = Networks.register(decorator(
        type(klass_name, (klass_map[klass_base_name], ), klass_params)))


CFG = {
    "VGG11": {
        "cfg": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,
                "M"]
    },
    "VGG13": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512,
                512, "M"]
    },
    "VGG16": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512,
                "M", 512, 512, 512, "M"]
    },
    "VGG19": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512,
                512, 512, "M", 512, 512, 512, 512, "M"]
    },
    "DenseNet121": {
        "growth_rate": 32,
        "block_config": (6, 12, 24, 16),
        "num_init_features": 64
    },
    "DenseNet161": {
        "growth_rate": 48,
        "block_config": (6, 12, 36, 24),
        "num_init_features": 96
    },
    "DenseNet169": {
        "growth_rate": 32,
        "block_config": (6, 12, 32, 32),
        "num_init_features": 64
    },
    "DenseNet201": {
        "growth_rate": 32,
        "block_config": (6, 12, 48, 32),
        "num_init_features": 64
    },
    "ResNet18": {
        "block": models.resnet.BasicBlock,
        "layers": [2, 2, 2, 2]
    },
    "ResNet34": {
        "block": models.resnet.BasicBlock,
        "layers": [3, 4, 6, 3]
    },
    "ResNet50x32x4d": {
        "block": models.resnet.Bottleneck,
        "layers": [3, 4, 6, 3],
        "groups": 32,
        "width_per_group": 4
    },
    "ResNet50Wide": {
        "block": models.resnet.Bottleneck,
        "layers": [3, 4, 6, 3],
        "width_per_group": 64 * 2
    },
    "ResNet50": {
        "block": models.resnet.Bottleneck,
        "layers": [3, 4, 6, 3]
    },
    "ResNet101": {
        "block": models.resnet.Bottleneck,
        "layers": [3, 4, 23, 3]
    },
    "ResNet101x32x8d": {
        "block": models.resnet.Bottleneck,
        "layers": [3, 4, 23, 3],
        "groups": 32,
        "width_per_group": 8
    },
    "ResNet101Wide": {
        "block": models.resnet.Bottleneck,
        "layers": [3, 4, 23, 3],
        "width_per_group": 64 * 2
    },
    "ResNet152": {
        "block": models.resnet.Bottleneck,
        "layers": [3, 8, 36, 3]
    },

}


destination_module_globals = globals()
for klass_name, klass_params in CFG.items():
    class_factory(klass_name, klass_params, destination_module_globals)

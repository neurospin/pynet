# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Define classifier models.
"""

# System import
import re

# Third party import
from torchvision import models
import torch
import torch.nn.functional as func
import numpy as np

# Package import
from pynet.core import Base


class Classifier(Base):
    """ Class to perform classification.
    """
    def __init__(self, optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False,
                 pretrained=None, **kwargs):
        """ Class instantiation.

        Observers will be notified, allowed signals are:
        - 'before_epoch'
        - 'after_epoch'

        Parameters
        ----------
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
        pretrained: path, default None
            path to the pretrained model or weights.
        kwargs: dict
            specify directly a custom 'model', 'optimizer' or 'loss'. Can also
            be used to set specific optimizer parameters.
        """
        super().__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)


class VGGClassifier(Classifier):
    """ VGGNet (2014) by Simonyan and Zisserman.
    """
    def __init__(self, cfg, num_classes, batch_norm=False, init_weights=True,
                 pretrained=None, make_layers=models.vgg.make_layers,
                 optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False, **kwargs):
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
        self.model = models.VGG(
            features=make_layers(cfg, batch_norm=batch_norm),
            num_classes=num_classes,
            init_weights=init_weights)
        super().__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)


class DenseNetClassifier(Classifier):
    """ DenseNet.
    """
    def __init__(self, growth_rate, block_config, num_init_features,
                 num_classes, bn_size=4, drop_rate=0, memory_efficient=False,
                 pretrained=None, optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 use_cuda=False, **kwargs):
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
        pretrained: str, default None
            update the weights of the model using this state information.
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
        self.model = models.DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            num_classes=num_classes,
            memory_efficient=memory_efficient)
        super().__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)


class ResNetClassifier(Classifier):
    """ Residual Neural Network (ResNet) by Kaiming He et al.
    """
    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None,
                 pretrained=None, optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False, **kwargs):
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
        pretrained: str, default None
            update the weights of the model using this state information.
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
        self.model = models.ResNet(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer)
        super().__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)


class Inception3(Classifier):
    """ Inception v3 by Google.
    """
    def __init__(self, num_classes, aux_logits=True, transform_input=False,
                 pretrained=None, optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 use_cuda=False, **kwargs):
        """ Class initilization.

        Parameters
        ----------
        num_classes: int
            number of classification classes.
        aux_logits: bool, default False
            auxiliary classifier for the training.
        transform_input: bool, default False
            normalize the data.
        pretrained: str, default None
            update the weights of the model using this state information.
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
        self.model = models.Inception3(
            num_classes=num_classes,
            aux_logits=aux_logits,
            transform_input=transform_input)
        super().__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            use_cuda=use_cuda,
            pretrained=pretrained,
            **kwargs)


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
    class VGGBaseClassifier(VGGClassifier):
        """ VGGNet X-layer.
        """
        cfg = None

        def __init__(self, num_classes, batch_norm=False, init_weights=True,
                     pretrained=None, optimizer_name="Adam",
                     learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                     **kwargs):
            if self.cfg is None:
                raise ValueError("Please specify a configuration first.")
            super().__init__(
                cfg=self.cfg,
                num_classes=num_classes,
                batch_norm=batch_norm,
                init_weights=init_weights,
                pretrained=pretrained,
                make_layers=models.vgg.make_layers,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                loss_name=loss_name,
                metrics=metrics,
                **kwargs)

    class DenseNetBaseClassifier(DenseNetClassifier):
        """ DenseNet-X model.
        """
        growth_rate = None
        block_config = None
        num_init_features = None

        def __init__(self, num_classes, bn_size=4, drop_rate=0,
                     memory_efficient=False, pretrained=None,
                     optimizer_name="Adam", learning_rate=1e-3,
                     loss_name="NLLLoss", metrics=None, use_cuda=False,
                     **kwargs):
            for name in ("growth_rate", "block_config", "num_init_features"):
                if getattr(self, name) is None:
                    raise ValueError(
                        "Please specify '{0}' first.".format(name))
            super().__init__(
                growth_rate=self.growth_rate,
                block_config=self.block_config,
                num_init_features=self.num_init_features,
                num_classes=num_classes,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                pretrained=pretrained,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                loss_name=loss_name,
                metrics=metrics,
                **kwargs)

    class ResNetBaseClassifier(ResNetClassifier):
        """ ResNet-X model.
        """
        block = None
        layers = None
        groups = 1
        width_per_group = 64

        def __init__(self, num_classes, zero_init_residual=False, groups=1,
                     width_per_group=64, replace_stride_with_dilation=None,
                     norm_layer=None, pretrained=None,
                     optimizer_name="Adam", learning_rate=1e-3,
                     loss_name="NLLLoss", metrics=None, use_cuda=False,
                     **kwargs):
            for name in ("block", "layers"):
                if getattr(self, name) is None:
                    raise ValueError(
                        "Please specify '{0}' first.".format(name))
            super().__init__(
                block=self.block,
                layers=self.layers,
                num_classes=num_classes,
                zero_init_residual=zero_init_residual,
                groups=groups,
                width_per_group=width_per_group,
                replace_stride_with_dilation=replace_stride_with_dilation,
                norm_layer=norm_layer,
                pretrained=pretrained,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                loss_name=loss_name,
                metrics=metrics,
                **kwargs)
    klass_map = {
        "VGG": VGGBaseClassifier,
        "DenseNet": DenseNetBaseClassifier,
        "ResNet": ResNetBaseClassifier
    }
    klass_params.update({
        "__module__": destination_module_globals["__name__"],
        "_id":  destination_module_globals["__name__"] + "." + klass_name
    })
    klass_base_name = re.findall(r"([a-zA-Z]+)[0-9]+", klass_name)[0]
    destination_module_globals[klass_name] = (
        type(klass_name, (klass_map[klass_base_name], ), klass_params))


CFG = {
    "VGG11Classifier": {
        "cfg": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,
                "M"]
    },
    "VGG13Classifier": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512,
                512, "M"]
    },
    "VGG16Classifier": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512,
                "M", 512, 512, 512, "M"]
    },
    "VGG19Classifier": {
        "cfg": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512,
                512, 512, "M", 512, 512, 512, 512, "M"]
    },
    "DenseNet121Classifier": {
        "growth_rate": 32,
        "block_config": (6, 12, 24, 16),
        "num_init_features": 64
    },
    "DenseNet161Classifier": {
        "growth_rate": 48,
        "block_config": (6, 12, 36, 24),
        "num_init_features": 96
    },
    "DenseNet169Classifier": {
        "growth_rate": 32,
        "block_config": (6, 12, 32, 32),
        "num_init_features": 64
    },
    "DenseNet201Classifier": {
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

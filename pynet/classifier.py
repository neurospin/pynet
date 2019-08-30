# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import re
import warnings

# Third party import
from torchvision import models
import torch
import torch.nn.functional as func
import tqdm
import numpy as np
from sklearn.utils import gen_batches

# Package import
from pynet.utils import checkpoint
from pynet.history import History
import pynet.metrics as mmetrics


class Classifier(object):
    """ Class to perform classification.
    """    
    def __init__(self, batch_size="auto", optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 observers=None, use_cuda=False, **kwargs):
        """ Class instantiation.

        Parameters
        ----------
        batch_size: int, default 'auto'
            the mini-batches size.
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
        observers: list of callable
            a list of callable observers that will be notify after each epoch
            of the fit.
        use_cuda: bool, default False
            wether to use GPU or CPU.
        kwargs: dict
            specify directly a custom 'model', 'optimizer' or 'loss'. Can also
            be used to set specific optimizer parameters.
        """
        self.batch_size = batch_size
        self.optimizer = kwargs.get("optimizer")
        self.loss = kwargs.get("loss")
        for name in ("optimizer", "loss"):
            if name in kwargs:
                kwargs.pop(name)
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        if self.optimizer is None:
            if optimizer_name not in dir(torch.optim):
                raise ValueError("Optimizer '{0}' uknown: check available "
                                 "optimizer in 'pytorch.optim'.")
            self.optimizer = getattr(torch.optim, optimizer_name)(
                self.model.parameters(),
                lr=learning_rate,
                **kwargs)
        if self.loss is None:
            if loss_name not in dir(torch.nn):
                raise ValueError("Loss '{0}' uknown: check available loss in "
                                 "'pytorch.nn'.")
            self.loss = getattr(torch.nn, loss_name)()
        self.metrics = {}
        for name in (metrics or []):
            if name not in mmetrics.METRICS:
                raise ValueError("Metric '{0}' not yet supported: you can try "
                                 "to fill the 'METRICS' factory, or ask for "
                                 "some help!")
            self.metrics[name] = mmetrics.METRICS[name]
        self.observers = observers or []
        if use_cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: unset 'use_cuda' parameter.")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.model.to(self.device)

    def _gen_batches(self, n_samples):
        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn("Got 'batch_size' less than 1 or larger than "
                              "sample size. It is going to be clipped.")
            batch_size = np.clip(self.batch_size, 1, n_samples)
        batch_slices = list(gen_batches(n_samples, batch_size))
        return batch_slices

    def fit(self, X, y, nb_epochs=100, checkpointdir=None, fold=1):
        """ Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            the input data.
        y: array-like, shape (n_samples,) or (n_samples, n_outputs)
            the target values: class labels.
        nb_epochs: int, default 100
            the number of epochs.
        checkpointdir: str, default None
            a destination folder where intermediate outputs will be saved.
        fold: int, default 1
            the index of the current fold if applicable.

        Returns
        -------
        history: History
            the fit history.
        """
        batch_slices = self._gen_batches(X.shape[0])
        nb_batch = len(batch_slices)
        history = History(name="fit")
        self.model.train()
        print(self.loss)
        print(self.optimizer)
        for epoch in range(1, nb_epochs + 1):
            values = {}
            loss = 0
            trange = tqdm.trange(1, nb_batch + 1, desc="Batch")
            for iteration in trange:
                trange.set_description("Batch {0}".format(iteration))
                trange.refresh()
                batch_slice = batch_slices[iteration - 1]
                batch_X = torch.from_numpy(X[batch_slice]).to(self.device)
                batch_y = torch.from_numpy(y[batch_slice]).to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(batch_X)
                batch_loss = self.loss(y_pred, batch_y)
                batch_loss.backward()
                self.optimizer.step()
                loss += batch_loss.item()
                for name, metric in self.metrics.items():
                    if name not in values:
                        values[name] = 0
                    values[name] += metric(y_pred, batch_y) / nb_batch
            history.log((fold, epoch), loss=loss, **values)
            history.summary()
            if checkpointdir is not None:
                checkpoint(
                    model=self.model,
                    epoch=epoch,
                    fold=fold,
                    outdir=checkpointdir)
                history.save(outdir=checkpointdir, epoch=epoch, fold=fold)
            for observer in self.observers:
                observer(self, epoch=epoch, fold=fold)
        return history

    def predict_proba(self, X):
        """ Predict classes probabilities using the defined classifier network.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            the input data.

        Returns
        -------
        y_probs: array-like, shape (n_samples,) or (n_samples, n_classes)
            the predicted classes associated probabilities.
        """
        batch_slices = self._gen_batches(X.shape[0])
        nb_batch = len(batch_slices)
        self.model.eval()
        with torch.no_grad():
            trange = tqdm.trange(1, nb_batch + 1, desc="Batch")
            y = []
            for iteration in trange:
                trange.set_description("Batch {0}".format(iteration))
                trange.refresh()
                batch_slice = batch_slices[iteration - 1]
                batch_X = torch.from_numpy(X[batch_slice]).to(self.device)
                y.append(self.model(batch_X))
            y = torch.cat(y, 0)
        y_probs = func.softmax(y, dim=1)
        return y_probs.cpu().detach().numpy()

    def predict(self, X):
        """ Predict classes using the defined classifier network.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            the input data.

        Returns
        -------
        y: array-like, shape (n_samples,) or (n_samples, n_classes)
            the predicted classes.
        """
        y_probs = self.predict_proba(X)
        return np.argmax(y_probs, axis=1)


class VGGClassifier(Classifier):
    """ VGGNet (2014) by Simonyan and Zisserman.
    """
    def __init__(self, cfg, num_classes, batch_norm=False, init_weights=True,
                 pretrained=None, make_layers=models.vgg.make_layers,
                 batch_size="auto", optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, observers=None,
                 use_cuda=False, **kwargs):
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
        batch_size: int, default 'auto'
            the mini-batches size.
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
        observers: list of callable
            a list of callable observers that will be notify after each epoch
            of the fit.
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
        if pretrained is not None:
            self.model.load_state_dict(torch.load(pretrained))
        super().__init__(
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            observers=observers,
            use_cuda=use_cuda,
            **kwargs)


class DenseNetClassifier(Classifier):
    """ DenseNet.
    """
    def __init__(self, growth_rate, block_config, num_init_features,
                 num_classes, bn_size=4, drop_rate=0, memory_efficient=False,
                 pretrained=None, batch_size="auto", optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 observers=None, use_cuda=False, **kwargs):
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
        batch_size: int, default 'auto'
            the mini-batches size.
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
        observers: list of callable
            a list of callable observers that will be notify after each epoch
            of the fit.
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
        if pretrained is not None:
            self.model.load_state_dict(torch.load(pretrained))
        super().__init__(
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            observers=observers,
            use_cuda=use_cuda,
            **kwargs)


class ResNetClassifier(Classifier):
    """ Residual Neural Network (ResNet) by Kaiming He et al.
    """
    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=None, batch_size="auto",
                 optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, observers=None,
                 use_cuda=False, **kwargs):
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
        batch_size: int, default 'auto'
            the mini-batches size.
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
        observers: list of callable
            a list of callable observers that will be notify after each epoch
            of the fit.
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
        if pretrained is not None:
            self.model.load_state_dict(torch.load(pretrained))
        super().__init__(
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            observers=observers,
            use_cuda=use_cuda,
            **kwargs)


class Inception3(Classifier):
    """ Inception v3 by Google.
    """
    def __init__(self, num_classes, aux_logits=True, transform_input=False,
                 pretrained=None, batch_size="auto", optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 observers=None, use_cuda=False, **kwargs):
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
        batch_size: int, default 'auto'
            the mini-batches size.
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
        observers: list of callable
            a list of callable observers that will be notify after each epoch
            of the fit.
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
        if pretrained is not None:
            self.model.load_state_dict(torch.load(pretrained))
        super().__init__(
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            loss_name=loss_name,
            metrics=metrics,
            observers=observers,
            use_cuda=use_cuda,
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
                     pretrained=None, batch_size="auto", optimizer_name="Adam",
                     learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                     observers=None, **kwargs):
            if self.cfg is None:
                raise ValueError("Please specify a configuration first.")
            super().__init__(
                cfg=self.cfg,
                num_classes=num_classes,
                batch_norm=batch_norm,
                init_weights=init_weights,
                pretrained=pretrained,
                make_layers=models.vgg.make_layers,
                batch_size=batch_size,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                loss_name=loss_name,
                metrics=metrics,
                observers=observers,
                **kwargs)
    class DenseNetBaseClassifier(DenseNetClassifier):
        """ DenseNet-X model.
        """
        growth_rate = None
        block_config = None
        num_init_features = None
        def __init__(self, num_classes, bn_size=4, drop_rate=0,
                     memory_efficient=False, pretrained=None, batch_size="auto",
                     optimizer_name="Adam", learning_rate=1e-3,
                     loss_name="NLLLoss", metrics=None, observers=None,
                     use_cuda=False, **kwargs):
            for name in ("growth_rate", "block_config", "num_init_features"):
                if getattr(self, name) is None:
                    raise ValueError("Please specify '{0}' first.".format(name))
            super().__init__(
                growth_rate=self.growth_rate,
                block_config=self.block_config,
                num_init_features=self.num_init_features,
                num_classes=num_classes,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                pretrained=pretrained,
                batch_size=batch_size,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                loss_name=loss_name,
                metrics=metrics,
                observers=observers,
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
                     norm_layer=None, pretrained=None, batch_size="auto",
                     optimizer_name="Adam", learning_rate=1e-3,
                     loss_name="NLLLoss", metrics=None, observers=None,
                     use_cuda=False, **kwargs):
            for name in ("block", "layers"):
                if getattr(self, name) is None:
                    raise ValueError("Please specify '{0}' first.".format(name))
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
                batch_size=batch_size,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                loss_name=loss_name,
                metrics=metrics,
                observers=observers,
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
        

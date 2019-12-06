# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Core classes.
"""

# System import
import re
import os
import warnings
from collections import OrderedDict

# Third party import
from torchvision import models
import torch
import torch.nn.functional as func
from tqdm import tqdm
import numpy as np

# Package import
from pynet.utils import checkpoint
from pynet.history import History
from pynet.observable import Observable
import pynet.metrics as mmetrics
from pynet.utils import reset_weights


class Base(Observable):
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
            signals=["before_epoch", "after_epoch"])
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
        if use_cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: unset 'use_cuda' parameter.")
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            if hasattr(checkpoint, "state_dict"):
                self.model.load_state_dict(checkpoint.state_dict())
            elif isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    self.model.load_state_dict(checkpoint["model"])
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                self.model.load_state_dict(checkpoint)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.model.to(self.device)

    def training(self, manager, nb_epochs, checkpointdir=None, fold_index=None,
                 scheduler=None, with_validation=True):
        """ Train the model.

        Parameters
        ----------
        manager: a pynet DataManager
            a manager containing the train and validation data.
        nb_epochs: int, default 100
            the number of epochs.
        checkpointdir: str, default None
            a destination folder where intermediate models/historues will be
            saved.
        fold_index: int, default None
            the index of the fold to use for the training, default use all the
            available folds.
        scheduler: torch.optim.lr_scheduler, default None
            a scheduler used to reduce the learning rate.
        with_validation: bool, default True
            if set use the validation dataset.

        Returns
        -------
        train_history, valid_history: History
            the train/validation history.
        """
        if not os.path.isdir(checkpointdir):
            os.mkdir(checkpointdir)
        train_history = History(name="train")
        if with_validation is not None:
            valid_history = History(name="validation")
        else:
            valid_history = None
        print(self.loss)
        print(self.optimizer)
        folds = range(manager.number_of_folds)
        if fold_index is not None:
            folds = [fold_index]
        for fold in folds:
            reset_weights(self.model)
            loaders = manager.get_dataloader(
                train=True,
                validation=True,
                fold_index=fold)
            for epoch in range(nb_epochs):
                self.notify_observers("before_epoch", epoch=epoch, fold=fold)
                loss, values = self.train(loaders.train)
                if scheduler is not None:
                    scheduler.step(loss)
                train_history.log((fold, epoch), loss=loss, **values)
                train_history.summary()
                if checkpointdir is not None:
                    checkpoint(
                        model=self.model,
                        epoch=epoch,
                        fold=fold,
                        outdir=checkpointdir,
                        optimizer=self.optimizer)
                    train_history.save(
                        outdir=checkpointdir,
                        epoch=epoch,
                        fold=fold)
                if with_validation:
                    _, loss, values = self.test(loaders.validation)
                    valid_history.log((fold, epoch), loss=loss, **values)
                    valid_history.summary()
                    if checkpointdir is not None:
                        valid_history.save(
                            outdir=checkpointdir,
                            epoch=epoch,
                            fold=fold)
                self.notify_observers("after_epoch", epoch=epoch, fold=fold)
        return train_history, valid_history

    def train(self, loader):
        """ Train the model on the trained data.

        Parameters
        ----------
        loader: a pytorch Dataset
            the data laoder.

        Returns
        -------
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """
        self.model.train()
        nb_batch = len(loader)
        values = {}
        loss = 0
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")
        for iteration, dataitem in enumerate(loader):
            pbar.update()
            inputs = dataitem.inputs.to(self.device)
            targets = []
            for item in (dataitem.outputs, dataitem.labels):
                if item is not None:
                    targets.append(item.to(self.device))
            if len(targets) == 1:
                targets = targets[0]
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            batch_loss = self.loss(outputs, targets)
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss.item() / nb_batch
            for name, metric in self.metrics.items():
                if name not in values:
                    values[name] = 0
                values[name] += metric(outputs, targets) / nb_batch
        pbar.close()
        return loss, values

    def testing(self, manager, with_logit=False, predict=False):
        """ Evaluate the model.

        Parameters
        ----------
        manager: a pynet DataManager
            a manager containing the test data.
        with_logit: bool, default False
            apply a softmax to the result.
        predict: bool, default False
            take the argmax over the channels.

        Returns
        -------
        y: array-like
            the predicted data.
        X: array-like
            the input data.
        y_true: array-like
            the true data if available.
        loss: float
            the value of the loss function if true data availble.
        values: dict
            the values of the metrics if true data availble.
        """
        loaders = manager.get_dataloader(test=True)
        y, loss, values = self.test(
            loaders.test, with_logit=with_logit, predict=predict)
        if loss == 0:
            loss, values, y_true = (None, None, None)
        else:
            y_true = []
            X = []
            targets = OrderedDict()
            for dataitem in loaders.test:
                for cnt, item in enumerate((dataitem.outputs,
                                            dataitem.labels)):
                    if item is not None:
                        targets.setdefault(cnt, []).append(
                            item.cpu().detach().numpy())
                X.append(dataitem.inputs.cpu().detach().numpy())
            X = np.concatenate(X, axis=0)
            for key, values in targets.items():
                y_true.append(np.concatenate(values, axis=0))
            if len(y_true) == 1:
                y_true = y_true[0]
        return y, X, y_true, loss, values

    def test(self, loader, with_logit=False, predict=False):
        """ Evaluate the model on the test or validation data.

        Parameters
        ----------
        loader: a pytorch Dataset
            the data laoder.
        with_logit: bool, default False
            apply a softmax to the result.
        predict: bool, default False
            take the argmax over the channels.

        Returns
        -------
        y: array-like
            the predicted data.
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """
        self.model.eval()
        nb_batch = len(loader)
        loss = 0
        values = {}
        with torch.no_grad():
            y = []
            pbar = tqdm(total=nb_batch, desc="Mini-Batch")
            for iteration, dataitem in enumerate(loader):
                pbar.update()
                inputs = dataitem.inputs.to(self.device)
                targets = []
                for item in (dataitem.outputs, dataitem.labels):
                    if item is not None:
                        targets.append(item.to(self.device))
                if len(targets) == 1:
                    targets = targets[0]
                elif len(targers) == 0:
                    targets = None
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    y.append(outputs[0])
                else:
                    y.append(outputs)
                if targets is not None:
                    batch_loss = self.loss(outputs, targets)
                    loss += float(batch_loss) / nb_batch
                    for name, metric in self.metrics.items():
                        if name not in values:
                            values[name] = 0
                        values[name] += metric(outputs, targets) / nb_batch
            pbar.close()
            y = torch.cat(y, 0)
            if with_logit:
                y = func.softmax(y, dim=1)
            y = y.cpu().detach().numpy()
            if predict:
                y = np.argmax(y, axis=1)
        return y, loss, values

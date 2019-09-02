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
from pynet.observable import Observable
import pynet.metrics as mmetrics


class Base(Observable):
    """ Class to perform classification.
    """    
    def __init__(self, batch_size="auto", optimizer_name="Adam",
                 learning_rate=1e-3, loss_name="NLLLoss", metrics=None,
                 use_cuda=False, **kwargs):
        """ Class instantiation.

        Observers will be notified, allowed signals are:
        - 'before_epoch'
        - 'after_epoch'        

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
        use_cuda: bool, default False
            wether to use GPU or CPU.
        kwargs: dict
            specify directly a custom 'model', 'optimizer' or 'loss'. Can also
            be used to set specific optimizer parameters.
        """
        super().__init__(
            signals=["before_epoch", "after_epoch"])
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
            self.notify_observers("before_epoch", epoch=epoch, fold=fold)
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
            self.notify_observers("after_epoch", epoch=epoch, fold=fold)
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

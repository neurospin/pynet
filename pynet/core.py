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
import copy
import types
import warnings
import logging
from collections import OrderedDict

# Third party import
import torch
import torch.nn.functional as func
import progressbar
import numpy as np

# Package import
from pynet.utils import checkpoint
from pynet.history import History
from pynet.observable import Observable
from pynet.utils import Metrics


# Global parameters
logger = logging.getLogger("pynet")


class Base(Observable):
    """ Base class for perform Deep Learning training.
    """
    def __init__(self, optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False,
                 pretrained=None, resume=False, add_labels=False, **kwargs):
        """ Class instantiation.

        Observers will be notified, allowed signals are:
        - 'before_epoch'
        - 'after_epoch'
        - 'kernel_regularizer'

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
        resume: bool, default False
            if set to true, the code will restore the weights of the model
            but also restore the optimizer's state, as well as the
            hyperparameters used, and the scheduler.
        add_labels: bool, default False
            if set and labels are specified in the data manager, add the labels
            to the forward function parameters.
        kwargs: dict
            specify directly a custom 'model', 'optimizer' or 'loss'. Can also
            be used to set specific optimizer parameters.
        """
        super().__init__(
            signals=["before_epoch", "after_epoch", "regularizer"])
        self.optimizer = kwargs.get("optimizer")
        self.loss = kwargs.get("loss")
        self.resume = resume
        self.add_labels = add_labels
        for name in ("optimizer", "loss"):
            if name in kwargs:
                kwargs.pop(name)
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        if self.optimizer is None:
            if optimizer_name not in dir(torch.optim):
                raise ValueError(
                    "Optimizer '{0}' uknown: check available optimizer in "
                    "'pytorch.optim'.".format(optimizer_name))
            self.optimizer = getattr(torch.optim, optimizer_name)(
                self.model.parameters(),
                lr=learning_rate,
                **kwargs)
        if self.loss is None:
            if loss_name not in dir(torch.nn):
                raise ValueError("Loss '{0}' unknown: check available loss in "
                                 "'pytorch.nn'.".format(loss_name))
            self.loss = getattr(torch.nn, loss_name)()
        self.metrics = {}
        for obj_or_name in (metrics or []):
            if isinstance(obj_or_name, types.FunctionType):
                self.metrics[obj_or_name.__name__] = obj_or_name
                continue
            if hasattr(obj_or_name, "__call__"):
                self.metrics[obj_or_name.__class__.__name__] = obj_or_name
                continue
            if obj_or_name not in Metrics.get_registry():
                logger.info("Available metrics:\n{0}".format(
                    Metrics.get_registry()))
                raise ValueError("Metric '{0}' not yet supported: you can try "
                                 "to fill the 'Metrics' factory, or ask for "
                                 "some help!".format(obj_or_name))
            self.metrics[obj_or_name] = Metrics.get_registry()[obj_or_name]
        if use_cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: unset 'use_cuda' parameter.")
        self.checkpoint = None
        if pretrained is not None:
            kwargs = {}
            if not use_cuda:
                kwargs["map_location"] = torch.device("cpu")
            self.checkpoint = torch.load(pretrained, **kwargs)
            if hasattr(self.checkpoint, "state_dict"):
                self.model.load_state_dict(self.checkpoint.state_dict())
            elif isinstance(self.checkpoint, dict):
                if "model" in self.checkpoint:
                    self.model.load_state_dict(self.checkpoint["model"])
                if self.resume and "optimizer" in self.checkpoint:
                    self.optimizer.load_state_dict(
                        self.checkpoint["optimizer"])
            else:
                self.model.load_state_dict(self.checkpoint)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.model.to(self.device)

    def training(self, manager, nb_epochs, checkpointdir=None, fold_index=None,
                 scheduler=None, with_validation=True, save_after_epochs=1,
                 add_labels=False):
        """ Train the model.

        Parameters
        ----------
        manager: a pynet DataManager
            a manager containing the train and validation data.
        nb_epochs: int, default 100
            the number of epochs.
        checkpointdir: str, default None
            a destination folder where intermediate models/histories will be
            saved.
        fold_index: int, default None
            the index of the fold to use for the training, default use all the
            available folds.
        scheduler: torch.optim.lr_scheduler, default None
            a scheduler used to reduce the learning rate.
        with_validation: bool, default True
            if set use the validation dataset.
        save_after_epochs: int, default 1
            determines when the model is saved and represents the number of
            epochs before saving.

        Returns
        -------
        train_history, valid_history: History
            the train/validation history.
        """
        if self.resume and "scheduler" in self.checkpoint:
            scheduler.load_state_dict(self.checkpoint["scheduler"])
        if checkpointdir is not None and not os.path.isdir(checkpointdir):
            os.mkdir(checkpointdir)
        train_history = History(name="train")
        if with_validation is not None:
            valid_history = History(name="validation")
        else:
            valid_history = None
        logger.info("Loss function {0}.".format(self.loss))
        logger.info("Optimizer function {0}.".format(self.optimizer))
        folds = range(manager.number_of_folds)
        if fold_index is not None:
            folds = [fold_index]
        init_optim_state = copy.deepcopy(self.optimizer.state_dict())
        init_model_state = copy.deepcopy(self.model.state_dict())
        if scheduler is not None:
            init_scheduler_state = copy.deepcopy(scheduler.state_dict())
        for fold in folds:
            logger.debug("Running fold {0}...".format(fold))
            self.optimizer.load_state_dict(init_optim_state)
            self.model.load_state_dict(init_model_state)
            if scheduler is not None:
                scheduler.load_state_dict(init_scheduler_state)
            loaders = manager.get_dataloader(
                train=True,
                validation=with_validation,
                fold_index=fold)
            for epoch in range(nb_epochs):
                logger.debug("Running epoch {0}:".format(fold))
                logger.debug("  notify observers with signal 'before_epoch'.")
                self.notify_observers("before_epoch", epoch=epoch, fold=fold,
                                      scheduler=scheduler)
                observers_kwargs = {}
                logger.debug("  train.")
                loss, values = self.train(loaders.train)
                observers_kwargs["loss"] = loss
                observers_kwargs.update(values)
                if scheduler is not None:
                    logger.debug("  update scheduler.")
                    scheduler.step(loss)
                    if hasattr(scheduler, "get_last_lr"):
                        logger.debug("  - lr: {0}".format(
                            scheduler.get_last_lr()))
                    else:
                        logger.debug("  - lr: {0}".format(scheduler._last_lr))
                logger.debug("  update train history.")
                train_history.log((fold, epoch), loss=loss, **values)
                train_history.summary()
                if (checkpointdir is not None and
                        epoch % save_after_epochs == 0):
                    logger.debug("  create checkpoint.")
                    checkpoint(
                        model=self.model,
                        epoch=epoch,
                        fold=fold,
                        outdir=checkpointdir,
                        optimizer=self.optimizer,
                        scheduler=scheduler)
                    train_history.save(
                        outdir=checkpointdir,
                        epoch=epoch,
                        fold=fold)
                if with_validation:
                    logger.debug("  validation.")
                    y_pred, loss, values = self.test(
                        loaders.validation, is_validation=True)
                    observers_kwargs["val_loss"] = loss
                    observers_kwargs.update(dict(
                        ("val_{0}".format(key), val)
                        for key, val in values.items()))
                    observers_kwargs["val_pred"] = y_pred
                    logger.debug("  update validation history.")
                    valid_history.log((fold, epoch), loss=loss, **values)
                    valid_history.summary()
                    if (checkpointdir is not None and
                            epoch % save_after_epochs == 0):
                        logger.debug("  create checkpoint.")
                        valid_history.save(
                            outdir=checkpointdir,
                            epoch=epoch,
                            fold=fold)
                logger.debug("  notify observers with signal 'after_epoch'.")
                self.notify_observers("after_epoch", epoch=epoch, fold=fold,
                                      scheduler=scheduler, **observers_kwargs)
                logger.debug("End epoch.".format(fold))
            logger.debug("End fold.")
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
        logger.debug("Update model for training.")
        self.model.train()
        nb_batch = len(loader)
        values = {}
        loss = 0
        pbar = progressbar.ProgressBar(
            max_value=nb_batch, redirect_stdout=True, prefix="Mini-batch ")
        pbar.start()
        for iteration, dataitem in enumerate(loader):
            logger.debug("Mini-batch {0}:".format(iteration))
            pbar.update(iteration + 1)
            logger.debug("  transfer inputs to {0}.".format(self.device))
            inputs = self._to_device(dataitem.inputs)
            logger.debug("  transfer targets to {0}.".format(self.device))
            targets = []
            for item in (dataitem.outputs, dataitem.labels):
                if item is not None:
                    targets.append(self._to_device(item))
            args = ()
            if self.add_labels and dataitem.labels is not None:
                args = (targets[-1], )
            logger.debug("  evaluate model.")
            self.optimizer.zero_grad()
            output_items = self.model(inputs, *args)
            outputs, layer_outputs = self._parse_outputs(output_items)
            logger.debug("  update loss.")
            if isinstance(outputs, list):
                logger.debug("  outputs: {0}".format(len(outputs)))
            elif isinstance(outputs, np.ndarray):
                logger.debug("  outputs: {0} - {1}".format(
                    outputs.shape, outputs.dtype))
            logger.debug("  targets: {0}".format(len(targets)))
            if hasattr(self.loss, "layer_outputs"):
                self.loss.layer_outputs = layer_outputs
            batch_items = self.loss(outputs, *targets)
            batch_loss, batch_loss_metrics = self._parse_outputs(batch_items)
            if batch_loss_metrics is not None:
                for name, metric in batch_loss_metrics.items():
                    if name not in values:
                        values[name] = 0
                    values[name] += float(metric) / nb_batch
            regularizations = self.notify_observers(
                "regularizer", layer_outputs=layer_outputs)
            for reg in regularizations:
                batch_loss += reg
            logger.debug("  update model weights.")
            batch_loss.backward()
            self.optimizer.step()
            loss += batch_loss.item() / nb_batch
            for name, metric in self.metrics.items():
                logger.debug("  compute metric '{0}'.".format(name))
                if hasattr(metric, "layer_outputs"):
                    metric.layer_outputs = layer_outputs
                if name not in values:
                    values[name] = 0
                values[name] += float(metric(outputs, *targets)) / nb_batch
            logger.debug("Mini-batch done.")
        pbar.finish()
        logger.debug("Loss {0} ({1})".format(loss, type(loss)))
        return loss, values

    def testing(self, manager, with_logit=False, logit_function="softmax",
                predict=False, concat_layer_outputs=None):
        """ Evaluate the model.

        Parameters
        ----------
        manager: a pynet DataManager
            a manager containing the test data.
        with_logit: bool, default False
            apply the logit function to the result.
        logit_function: str, default 'softmax'
            choose the logit function.
        predict: bool, default False
            take the argmax over the channels.
        concat_layer_outputs: list of str, default None
            the outputs of the intermediate layers to be merged with the
            predicted data (must be the same size).

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
            loaders.test, with_logit=with_logit, logit_function=logit_function,
            predict=predict, concat_layer_outputs=concat_layer_outputs)
        if loss == 0:
            loss, values, y_true, X = (None, None, None, None)
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
            for key, _values in targets.items():
                y_true.append(np.concatenate(_values, axis=0))
            if len(y_true) == 1:
                y_true = y_true[0]
        return y, X, y_true, loss, values

    def test(self, loader, with_logit=False, logit_function="softmax",
             predict=False, concat_layer_outputs=None, is_validation=False):
        """ Evaluate the model on the test or validation data.

        Parameters
        ----------
        loader: a pytorch Dataset
            the data laoder.
        with_logit: bool, default False
            apply the logit function to the result.
        logit_funtction: str, default 'softmax'
            choose the logit function.
        predict: bool, default False
            take the argmax over the channels.
        concat_layer_outputs: list of str, default None
            the outputs of the intermediate layers to be merged with the
            predicted data (must be the same size).
        is_validation: bool default False
            specify if we are in the validation phase.

        Returns
        -------
        y: array-like
            the predicted data.
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """
        logger.debug("Update model for testing.")
        self.model.eval()
        nb_batch = len(loader)
        loss = 0
        values = {}
        concate_out = True
        with torch.no_grad():
            y = []
            pbar = progressbar.ProgressBar(
                max_value=nb_batch, redirect_stdout=True, prefix="Mini-batch ")
            pbar.start()
            for iteration, dataitem in enumerate(loader):
                logger.debug("Mini-batch {0}:".format(iteration))
                pbar.update(iteration + 1)
                logger.debug("  transfer inputs to {0}.".format(self.device))
                inputs = self._to_device(dataitem.inputs)
                logger.debug("  transfer targets to {0}.".format(self.device))
                targets = []
                for item in (dataitem.outputs, dataitem.labels):
                    if item is not None:
                        targets.append(self._to_device(item))
                args = ()
                if self.add_labels and dataitem.labels is not None:
                    args = (targets[-1], )
                logger.debug("  evaluate model.")
                output_items = self.model(inputs, *args)
                extra_outputs = []
                outputs, layer_outputs = self._parse_outputs(output_items)
                if (concat_layer_outputs is not None and
                        layer_outputs is not None):
                    for name in concat_layer_outputs:
                        if name not in layer_outputs:
                            raise ValueError(
                                "Unknown layer output '{0}'. Check the "
                                "network forward method.".format(name))
                        extra_outputs.append(layer_outputs[name])
                if is_validation or len(targets) > 0:
                    logger.debug("  update loss.")
                    logger.debug("  layer outputs: {0}".format(layer_outputs))
                    if hasattr(self.loss, "layer_outputs"):
                        self.loss.layer_outputs = layer_outputs
                    batch_items = self.loss(outputs, *targets)
                    batch_loss, batch_loss_metrics = self._parse_outputs(
                        batch_items)
                    if batch_loss_metrics is not None:
                        for name, metric in batch_loss_metrics.items():
                            if name not in values:
                                values[name] = 0
                            values[name] += float(metric) / nb_batch
                    loss += float(batch_loss) / nb_batch
                    for name, metric in self.metrics.items():
                        logger.debug("  compute metric '{0}'.".format(name))
                        if hasattr(metric, "layer_outputs"):
                            metric.layer_outputs = layer_outputs
                        if name not in values:
                            values[name] = 0
                        values[name] += metric(outputs, *targets) / nb_batch
                if len(extra_outputs) > 0:
                    y.append(torch.cat([outputs] + extra_outputs, 1))
                else:
                    if isinstance(outputs, list):
                        concate_out = False
                    y.append(outputs)
                logger.debug("Mini-batch done.")
            pbar.finish()
            if concate_out:
                try:
                    y = torch.cat(y, 0)
                except:
                    warnings.warn("Impossible to concatenate y!")
            if with_logit:
                logger.debug("Apply logit.")
                if logit_function == "softmax":
                    y = func.softmax(y, dim=1)
                elif logit_function == "sigmoid":
                    y = torch.sigmoid(y)
                else:
                    raise ValueError("Unsupported logit function.")
            y = self._numpy(y)
            if predict:
                logger.debug("Apply predict.")
                y = np.argmax(y, axis=1)
        return y, loss, values

    def _numpy(self, data):
        """ Transfer data to cpu as array data.

        Parameters
        ----------
        data: tensor or list of tensor
            the data to transfer.

        Returns
        -------
        out:  array or list of array
            the transfered data.
        """
        if isinstance(data, list):
            return [self._numpy(tensor) for tensor in data]
        elif torch.is_tensor(data):
            return data.cpu().detach().numpy()
        else:
            return data

    def _parse_outputs(self, output_items):
        """ Parse function/method outputs.

        Parameters
        ----------
        output_items: object, list or tuple
            the data to be parsed.

        Returns
        -------
        out: object
            the first output.
        extra_out: dict
            other outputs.
        """
        if (not isinstance(output_items, tuple) and
                not isinstance(output_items, list)):
            out = output_items
            extra_out = None
        elif len(output_items) == 1:
            out = output_items[0]
            extra_out = None
        elif len(output_items) == 2:
            out, extra_out = output_items
        else:
            raise ValueError(
                "The function / method can only return one or "
                "two parameters: the main output (loss, forward result), "
                "and as an option specific outputs in a dictionary.")
        if extra_out is not None and not isinstance(extra_out, dict):
            raise ValueError("Specific outputs must be a dictionary.")
        return out, extra_out

    def _to_device(self, data):
        """ Transfer data to device.

        Parameters
        ----------
        data: tensor or list of tensor
            the data to transfer.

        Returns
        -------
        out:  tensor or list of tensor
            the transfered data.
        """
        if isinstance(data, list):
            return [tensor.to(self.device) for tensor in data]
        else:
            return data.to(self.device)

# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides common oberservers that can be called during the fit.
"""

# Third party import
import torch

# Package import
from .history import History


class PredictObserver(object):
    """ Class to display model loss and metrics on a dataset.
    """
    def __init__(self, X, y, name):
        """ Initialize the class.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            the input data.
        y: array-like, shape (n_samples,) or (n_samples, n_outputs)
            the target values.
        name: str
            a observer name.
        """
        self.X = X
        self.y = y
        self.name = name
    def __call__(self, model, epoch, fold):
        """

        Parameters
        ----------
        model:
            the trained model.
        epoch: int
            the epoch number.
        fold: int
            the fold index.
        """
        history = History(name=self.name)
        model.model.eval()
        with torch.no_grad():
            y_pred = model.predict_proba(self.X)
            _y_pred = torch.from_numpy(y_pred)
            _y = torch.from_numpy(self.y)
            loss = model.loss(_y_pred, _y)
            values = {}
            for name, metric in model.metrics.items():
                values[name] = metric(_y_pred, _y)
            history.log((fold, epoch), loss=loss, **values)
            history.summary()
        

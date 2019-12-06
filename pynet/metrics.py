# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define common metrics.
"""

# Third party import
import torch
import numpy as np
import torch.nn.functional as func


def accuracy(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    accuracy = y_pred.eq(y).sum().cpu().numpy() / y.size()[0]
    return accuracy


def _dice(y_pred, y):
    """ Binary dice indice adapted to pytorch tensors.
    """
    flat_y_pred = torch.flatten(y_pred)
    flat_y = torch.flatten(y)
    intersection = (flat_y_pred * flat_y).sum()
    return (2. * intersection + 1.) / (flat_y_pred.sum() + flat_y.sum() + 1.)


def multiclass_dice(y_pred, y):
    """ Extension of the dice to a n classes problem.
    """
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_pred = func.softmax(y_pred, dim=1)
    dice = 0.
    n_classes = y.shape[1]
    for cnt in range(n_classes):
        dice += _dice(y_pred[:, cnt], y[:, cnt])
    return dice / n_classes


METRICS = {
    "accuracy": accuracy,
    "multiclass_dice": multiclass_dice,
}

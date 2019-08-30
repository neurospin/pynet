# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


def accuracy(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    accuracy = y_pred.eq(y).sum().cpu().numpy() / y.size()[0]
    return accuracy


METRICS = {
    "accuracy": accuracy
}

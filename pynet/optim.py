# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides optimization utilities.
"""


# System mport
import os

# Third party import
import torch

# Package import
from .history import History


def training(model, dataset, nb_epochs, outdir=None, verbose=0):
    """ Train the model.

    Parameters
    ----------
    model:
        a classification, autoencoder, gan model to be trained.
    dataset: dict
        the test, train, and validation loaders.
    nb_epochs: int
        the number of iteration over the input dataset.
    outdir: str, default None
        a destination folder where intermediate outputs will be saved.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    test_history, train_history: History
        the optimization history.
    """
    # Cross validation loop
    nb_folds =  len(dataset["train"])
    for fold in range(1, nb_folds + 1):
        nb_batch = len(dataset["train"][fold - 1])
        for batch in range(nb_batch):
            batch_data = dataset["train"][fold - 1][batch]
            X_train = batch_data["inputs"]
            if batch_data["outputs"] is not None:
                y_train = batch_data["outputs"]
            else:
                y_train = batch_data["labels"]
            train_history = model.fit(
                X_train, y_train, nb_epochs=nb_epochs, checkpointdir=outdir,
                fold=fold)

    # Test
    test_history = History(name="test", verbose=verbose)
    X_test = dataset["test"][0]["inputs"]
    if batch_data["outputs"] is not None:
        y_test = dataset["test"][0]["outputs"]
    else:
        y_test = dataset["test"][0]["labels"]
    with torch.no_grad():
        y_pred = model.predict_proba(X_test)
        _y_pred = torch.from_numpy(y_pred)
        _y_test = torch.from_numpy(y_test)
        loss = model.loss(_y_pred, _y_test)
        values = {}
        for name, metric in model.metrics.items():
            values[name] = metric(_y_pred, _y_test)
    test_history.log((0, 0), loss=loss, **values)
    test_history.summary()
    if outdir is not None:
        test_history.save(outdir=outdir, epoch=0, fold=0)

    return test_history, train_history
      


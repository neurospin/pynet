# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides optimisation utilities.
"""


# System mport
import os

# Third party import
import torch
import tqdm

# Package import
from .utils import checkpoint
from .history import History


def training(net, dataset, optimizer, criterion, nb_epochs, metrics=None,
             use_cuda=False, outdir=None, verbose=0):
    """ Train the model.

    Parameters
    ----------
    net:
        the network.
    dataset: dict
        the test, train, and validation loaders.
    optimizer:
        the desired optimisation strategy.
    criterion:
        the metric evaluated at each iteration to drive the optimisation.
    metrics: dict
        some metrics evaluated at each iteration and saved in the history.
    nb_epochs: int
        the number of iteration over the input dataset.
    use_cuda: bool, default False
        wether to use the GPU or not.
    outdir: str, default None
        a destination folder where intermediate outputs will be saved.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    test_history, train_history, valid_history: History
        the optimisation history.
    """
    # Define the device
    if use_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    device = torch.device("cuda" if use_cuda else "cpu")
    net = net.to(device)

    # Cross validation loop
    train_history = History(name="train", verbose=verbose)
    valid_history = History(name="valid", verbose=verbose)
    if outdir is not None:
        checkpointdir = os.path.join(outdir, "checkpoints")
        if not os.path.isdir(checkpointdir):
            os.mkdir(checkpointdir)
        historydir = os.path.join(outdir, "history")
        if not os.path.isdir(historydir):
            os.mkdir(historydir)
    for fold in range(1, len(dataset["train"]) + 1):
        for epoch in range(1, nb_epochs + 1):
            loss, values = train(
                model=net,
                dataloader=dataset["train"][fold - 1],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                metrics=metrics,
                verbose=verbose)
            train_history.log((fold, epoch), loss=loss, **values)
            loss, values = test(
                model=net,
                dataloader=dataset["validation"][fold - 1],
                criterion=criterion,
                device=device,
                metrics=metrics,
                verbose=verbose)
            valid_history.log((fold, epoch), loss=loss, **values)
            if verbose > 0:
                train_history.summary()
                valid_history.summary()
            if outdir is not None:
                checkpoint(
                    model=net,
                    epoch=epoch,
                    fold=fold,
                    outdir=checkpointdir)
                train_history.save(outdir=historydir, epoch=epoch, fold=fold)
                valid_history.save(outdir=historydir, epoch=epoch, fold=fold)

    # Test
    test_history = History(name="test", verbose=verbose)
    loss, values = test(
        model=net,
        dataloader=dataset["test"],
        criterion=criterion,
        device=device,
        metrics=metrics,
        verbose=verbose)
    test_history.log((0, 0), loss=loss, **values)
    if verbose > 0:
        test_history.summary()
    if outdir is not None:
        test_history.save(outdir=historydir, epoch=0, fold=0)

    return test_history, train_history, valid_history
      

def train(model, dataloader, optimizer, criterion, device, metrics=None, verbose=0):
    metrics = metrics or {}
    values = {}
    loss = 0
    current_loss = None
    nb_batch = len(dataloader)
    trange = tqdm.trange(1, nb_batch + 1, desc="Batch")
    for iteration in trange:
        trange.set_description("Batch {0}".format(current_loss))
        trange.refresh()
        batch_data = dataloader[iteration - 1]
        x = batch_data["inputs"].to(device)
        if batch_data["outputs"] is not None:
            y = batch_data["outputs"].to(device)
        else:
            y = batch_data["labels"].to(device)
        optimizer.zero_grad()
        prediction = model(x)
        _loss = criterion(prediction, y)
        current_loss = _loss.item()
        loss += current_loss
        _loss.backward()
        optimizer.step()
        for name, metric in metrics.items():
            if name not in values:
                values[name] = 0
            values[name] += metric(prediction, y) / nb_batch
    return loss, values


def test(model, dataloader, criterion, device, metrics=None, verbose=0):
    metrics = metrics or {}
    values = {}
    loss = 0
    current_loss = None
    nb_batch = len(dataloader)
    with torch.no_grad():
        trange = tqdm.trange(1, nb_batch + 1, desc="Batch")
        for iteration in trange:
            trange.set_description("Batch {0}".format(current_loss))
            trange.refresh()
            batch_data = dataloader[iteration - 1]
            x = batch_data["inputs"].to(device)
            if batch_data["outputs"] is not None:
                y = batch_data["outputs"].to(device)
            else:
                y = batch_data["labels"].to(device)
            y_pred = model(x)
            _loss = criterion(y_pred, y)
            current_loss = _loss.item()
            loss += current_loss
            for name, metric in metrics.items():
                if name not in values:
                    values[name] = 0
                values[name] += metric(y_pred, y) / nb_batch
    return loss, values




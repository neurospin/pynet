# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides functions to load and split a dataset.
"""


import torch
from torch.utils.data import Dataset

import nibabel
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


def split_dataset(path, dataloader, batch_size, inputs, label, number_of_folds,
                  outputs=None, transforms=None, verbose=0):
    """ Split an input tabular dataset in test, train and validation.
    This function stratifies the data.

    Parameters
    ----------
    path: str
        the path to the tabular data that will be splited.
    dataloader: class
        a pytorch Dataset derviced class used to load the data in a mini-batch
        fashion.
    batch_size: int
        the size of each mini-batch.
    inputs: list of str
        the name of the column(s) containing the inputs
    label: str
        the name of the column containing the labels.
    number_of_folds: int
        the number of folds that will be used in the cross validation.
    outputs: list of str, default None
        the name of the column(s) containing the ouputs.
    transforms: list of callable
        transform the dataset using these functions: cropping, reshaping,
        ...
    verbose: int, default 0
        the verbosity level.

    Retunrs
    -------
    dataset: dict
        the test, train, and validation loaders.
    """
    if outputs is not None:
        columns = inputs + outputs + [label]
    else:
        columns = inputs + [label]
    df = pd.read_csv(path, sep="\t")
    arr = df[columns].values
    arr_optim, arr_test = train_test_split(
        arr,
        test_size=0.1,
        random_state=1,
        stratify=arr[:, -1])
    dataset = {}
    testloader = LoadDataset(
        dataset=pd.DataFrame(arr_test, columns=columns),
        inputs=inputs,
        outputs=outputs,
        label=label,
        batch_size=batch_size,
        transforms=transforms,
        verbose=verbose)
    dataset["test"] = testloader
    for fold_indx in range(number_of_folds):
        arr_train, arr_valid = train_test_split(
            arr_optim,
            test_size=0.25,
            shuffle=True,
            stratify=arr_optim[:, -1])
        trainloader = LoadDataset(
            dataset=pd.DataFrame(arr_train, columns=columns),
            inputs=inputs,
            outputs=outputs,
            label=label,
            batch_size=batch_size,
            transforms=transforms,
            verbose=verbose)
        validloader = LoadDataset(
            dataset=pd.DataFrame(arr_valid, columns=columns),
            inputs=inputs,
            outputs=outputs,
            label=label,
            batch_size=batch_size,
            transforms=transforms,
            verbose=verbose)
        dataset.setdefault("train", []).append(trainloader)
        dataset.setdefault("validation", []).append(validloader)
    return dataset


def dummy_dataset(nb_batch, batch_size, number_of_folds, shape, verbose=0):
    """ Create a dummy dataset.

    Parameters
    ----------
    nb_batch: int
        the number of batch.
    batch_size: int
        the size of each mini-batch.
    number_of_folds: int
        the number of folds that will be used in the cross validation.
    shape: nuplet
        the generated data shape.
    verbose: int, default 0
        the verbosity level.

    Retunrs
    -------
    dataset: dict
        the test, train, and validation loaders.
    """
    dataset = {
        "test": DummyDataset(
            nb_batch, batch_size, shape, verbose=verbose),
        "train": [DummyDataset(
            nb_batch, batch_size, shape, verbose=verbose)] * number_of_folds,
        "validation": [DummyDataset(
            nb_batch, batch_size, shape, verbose=verbose)] * number_of_folds}
    return dataset


class LoadDataset(Dataset):
    """ Class to load a dataset in a mini-batch fashion.

    Note: the image are expected to be in the FSL order X, Y, Z, N.
    """

    def __init__(self, dataset, inputs, label, outputs=None, batch_size=10,
                 transforms=None, verbose=0):
        """ Initialize the class.

        Parameters
        ----------
        dataset: DataFrame
            a pandas dataframe.
        inputs: list of str
            the name of the column(s) containing the inputs
        label: str
            the name of the column containing the labels.
        outputs: list of str, default None
            the name of the column(s) containing the ouputs.
        batch_size: int, default 10
            the size of each mini-batch.
        transforms: list of callable
            transform the dataset using these functions: cropping, reshaping,
            ...
        verbose: int, default 0
            the verbosity level.
        """
        self.transforms = transforms or []
        self.batch_size = batch_size
        self.verbose = verbose
        self.dataset = dataset
        self.inputs = inputs
        self.outputs = outputs
        self.label = label
        self.all_labels = sorted(np.unique(self.dataset[label].values))
        div, self.rest = divmod(len(self.dataset), self.batch_size)
        self.nb_batch = div
        if self.rest > 0:
            self.nb_batch += 1

    def __len__(self):
        """ Get the number of mini-batch.

        Returns
        -------
        nb_batch: int
            the number of mini-batchs.
        """
        return self.nb_batch

    def _concat_features(self, row, names):
        """ Concate the inputs or ouputs in a 1, NB_FEATURES, X, Y, Z array.
        """
        arr = None
        names = names or []
        for name in names:
            _arr = nibabel.load(row[name].values[0]).get_data()
            for trf in self.transforms:
                _arr = trf(_arr)
            if _arr.ndim == 3:
                _arr = np.expand_dims(_arr, axis=0)
            else:
                _arr = np.ascontiguousarray(_arr.transpose(3, 0, 1, 2))
            _arr = np.expand_dims(_arr, axis=0)
            if arr is None:
                arr = _arr
            else:
                arr = np.concatenate((arr, _arr), axis=1)
        return arr

    def __getitem__(self, index):
        """ Get the desired mini-batch.

        Parameters
        ----------
        index: int
            the index of the desired mini-batch.

        Returns
        -------
        data: dict
            a dictionary with the inputs, ouputs, and labels mini-batch arrays.
        """
        if index >= self.nb_batch:
            raise IndexError("Index '{0}' is nout of range.".format(index))
        if self.rest > 0 and index == self.nb_batch - 1:
            index_end = self.batch_size * index + self.rest
        else:
            index_end = self.batch_size * (index + 1)
        input_arr = None
        output_arr = None
        labels = []
        for _idx in range(self.batch_size * index, index_end):
            row = self.dataset.iloc[[_idx]]
            _input_arr = self._concat_features(row, self.inputs)
            _output_arr = self._concat_features(row, self.outputs)
            if input_arr is None:
                input_arr = _input_arr
                if _output_arr is not None:
                    output_arr = _output_arr
            else:
                input_arr = np.concatenate((input_arr, _input_arr), axis=0)
                if _output_arr is not None:
                    output_arr = np.concatenate(
                        (output_arr, _output_arr), axis=0)
            labels.append(row[self.label].values[0])
        labels = np.asarray(labels) - 1
        if self.verbose > 0:
            print("-" * 50)
            print("Mini Batch: ", index)
            print("Mini Batch Size: ", self.batch_size)
            print("Inputs: ", input_arr.shape)
            print("Ouputs: ", output_arr.shape
                              if output_arr is not None else None)
            print("Labels: ", labels.shape)
        data = {
            "inputs": torch.from_numpy(input_arr),
            "outputs": torch.from_numpy(output_arr)
                       if output_arr is not None else None,
            "labels": torch.from_numpy(labels)
        }

        return data


class DummyDataset(Dataset):
    """ Class to load a toy dataset in a mini-batch fashion.
    """

    def __init__(self, nb_batch, batch_size, shape, verbose=0):
        """ Initialize the class.

        Parameters
        ----------
        nb_batch: int
            the number of batch.
        batch_size: int
            the size of each mini-batch.
        outputs: list of str, default None
            the name of the column(s) containing the ouputs.
        shape: nuplet
            the generated data shape.
        verbose: int, default 0
            the verbosity level.
        """
        self.batch_size = batch_size
        self.shape = shape
        self.verbose = verbose
        self.nb_batch = nb_batch

    def __len__(self):
        """ Get the number of mini-batch.

        Returns
        -------
        nb_batch: int
            the number of mini-batchs.
        """
        return self.nb_batch

    def __getitem__(self, index):
        """ Get the desired mini-batch.

        Parameters
        ----------
        index: int
            the index of the desired mini-batch.

        Returns
        -------
        data: dict
            a dictionary with the inputs, ouputs, and labels mini-batch arrays.
        """
        shape = [self.batch_size, 1] + list(self.shape)
        input_arr = np.random.random(shape).astype(np.single)
        output_arr = np.zeros(shape, dtype=np.single)
        labels = np.zeros((self.batch_size, ), dtype=int)
        if self.verbose > 0:
            print("-" * 50)
            print("Mini Batch: ", index)
            print("Mini Batch Size: ", self.batch_size)
            print("Inputs: ", input_arr.shape)
            print("Ouputs: ", output_arr.shape)
            print("Labels: ", labels.shape)
        data = {
            "inputs": torch.from_numpy(input_arr),
            "outputs": torch.from_numpy(output_arr)
                       if output_arr is not None else None,
            "labels": torch.from_numpy(labels)
        }
        return data 


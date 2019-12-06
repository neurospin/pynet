# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides core functions to load and split a dataset.
"""

# Imports
from collections import namedtuple, OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
import progressbar
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)

# Global parameters
SetItem = namedtuple("SetItem", ["test", "train", "validation"])
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])


class DataManager(object):
    """ Data manager used to split a dataset in train, test and validation
    pytorch datasets.
    """

    def __init__(self, input_path, metadata_path, output_path=None,
                 labels=None, stratify_label=None, projection_labels=None,
                 number_of_folds=10, batch_size=1, input_transforms=None,
                 output_transforms=None, add_input=False, test_size=0.1,
                 **dataloader_kwargs):
        """ Splits an input numpy array using memory-mapping into three sets:
        test, train and validation. This function can stratify the data.

        TODO: add how validation split is perform.
        TODO: fix case number_of_folds=1

        Parameters
        ----------
        input_path: str
            the path to the numpy array containing the input tensor data
            that will be splited/loaded.
        metadata_path: str
            the path to the metadata table in tsv format.
        output_path: str, default None
            the path to the numpy array containing the output tensor data
            that will be splited/loaded.
        labels: list of str, default None
            in case of classification/regression, the name of the column(s)
            in the metadata table to be predicted.
        stratify_label: str, default None
            the name of the column in the metadata table containing the label
            used during the stratification.
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.
        number_of_folds: int, default 10
            the number of folds that will be used in the cross validation.
        batch_size: int, default 1
            the size of each mini-batch.
        input_transforms, output_transforms: list of callable, default None
            transforms a list of samples with pre-defined transformations.
        add_input: bool, default False
            if true concatenate the input tensor to the output tensor.
        test_size: float, default 0.1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split.
        """
        df = pd.read_csv(metadata_path, sep="\t")
        projected_index = DataManager.get_projected_index(
            df=df,
            projection_labels=projection_labels)
        self.inputs = np.load(input_path, mmap_mode='r')
        self.outputs, self.labels, self.stratify_label = (None, None, None)
        if output_path is not None:
            self.outputs = np.load(output_path, mmap_mode='r')
        if labels is not None:
            self.labels = df[labels].values
            self.labels = self.labels.squeeze()
        if stratify_label is not None:
            self.stratify_label = df[stratify_label].values
        self.number_of_folds = number_of_folds
        self.batch_size = batch_size
        self.input_transforms = input_transforms
        self.output_transforms = output_transforms
        self.add_input = add_input
        self.data_loader_kwargs = dataloader_kwargs
        self.dataset = dict((key, [])
                            for key in ("train", "test", "validation"))

        # 1st step: split into train/test (get only indices)
        indices = np.ones(len(projected_index))
        if stratify_label is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=1, random_state=0, test_size=test_size)
            train_index, test_index = next(
                splitter.split(indices,
                               self.stratify_label[projected_index]))
        else:
            if test_size == 1:
                train_index, test_index = (None, range(len(projected_index)))
            else:
                splitter = ShuffleSplit(
                    n_splits=1, random_state=0, test_size=test_size)
                train_index, test_index = next(
                    splitter.split(np.ones(len(projected_index))))
        self.dataset["test"] = ArrayDataset(
            self.inputs, projected_index[test_index], labels=self.labels,
            outputs=self.outputs, add_input=self.add_input)
        if train_index is None:
            return

        # 2nd step: split the training set into K folds (K-1 for training, 1
        # for validation, K times)
        if stratify_label is not None:
            kfold_splitter = StratifiedKFold(
                n_splits=number_of_folds, shuffle=True, random_state=0)
            gen = kfold_splitter.split(
                np.ones(len(train_index)),
                self.stratify_label[projected_index][train_index])
        else:
            kfold_splitter = KFold(
                n_splits=number_of_folds, shuffle=True, random_state=0)
            gen = kfold_splitter.split(np.ones(len(train_index)))
        for fold_train_index, fold_val_index in gen:
            train_dataset = ArrayDataset(
                self.inputs, projected_index[train_index[fold_train_index]],
                labels=self.labels, outputs=self.outputs,
                add_input=self.add_input)
            val_dataset = ArrayDataset(
                self.inputs, projected_index[train_index[fold_val_index]],
                labels=self.labels, outputs=self.outputs,
                add_input=self.add_input)
            self.dataset["train"].append(train_dataset)
            self.dataset["validation"].append(val_dataset)

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: Dataset or list of Dataset
            the requested set of data: test, train or validation.
        """
        if item not in ("train", "test", "validation"):
            raise ValueError("Unknown set! Must be 'train', 'test' or "
                             "'validation'.")
        return self.dataset[item]

    def collate_fn(self, list_samples):
        """ After fetching a list of samples using the indices from sampler,
        the function passed as the collate_fn argument is used to collate lists
        of samples into batches.

        A custom collate_fn is used here to apply the transformations.

        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
        """
        data = OrderedDict()
        for cnt, dataset in enumerate(list_samples):
            _inputs = dataset.inputs
            for tf in (self.input_transforms or []):
                _inputs = tf(_inputs)
            data.setdefault("inputs", []).append(
                torch.as_tensor(_inputs))
            if dataset.outputs is not None:
                _outputs = dataset.outputs
                for tf in (self.output_transforms or []):
                    _outputs = tf(_outputs)
                data.setdefault("outputs", []).append(
                    torch.as_tensor(_outputs))
            if dataset.labels is not None:
                data.setdefault("labels", []).append(
                    torch.as_tensor(dataset.labels))
        for key in ("inputs", "outputs", "labels"):
            if key not in data:
                data[key] = None
            else:
                data[key] = torch.stack(data[key], dim=0).float()
        if data["labels"] is not None:
            data["labels"] = data["labels"].type(torch.LongTensor)
        return DataItem(**data)

    def get_dataloader(self, train=False, validation=False, test=False,
                       fold_index=0):
        """ Generate a putorch DataLoader.

        Parameters
        ----------
        train: bool, default False
            return the dataloader over the train set.
        validation: bool, default False
            return the dataloader over the validation set.
        test: bool, default False
            return the dataloader over the test set.
        fold_index: int, default 0
            the index of the fold to use for the training

        Returns
        -------
        loaders: list of DataLoader
            the requested data loaders.
        """
        _test, _train, _validation = (None, None, None)
        if test:
            _test = DataLoader(
                self.dataset["test"], batch_size=self.batch_size,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if train:
            _train = DataLoader(
                self.dataset["train"][fold_index], batch_size=self.batch_size,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if validation:
            _validation = DataLoader(
                self.dataset["validation"][fold_index],
                batch_size=self.batch_size, collate_fn=self.collate_fn,
                **self.data_loader_kwargs)
        return SetItem(test=_test, train=_train, validation=_validation)

    @staticmethod
    def get_projected_index(df, projection_labels=None):
        """ Filter a table.

        Parameters
        ----------
        df: a pandas DataFrame
            a table data.
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.

        Returns
        -------
        projected_index: iterable with int
            the selected rows.
        """
        if projection_labels is None:
            return df.index
        projected_index = df.index
        for (col, val) in projection_labels.items():
            projected_index = projected_index.intersection(
                df[df[col] == val].index)
        return projected_index


class ArrayDataset(Dataset):
    """ A dataset based on numpy array.
    """
    def __init__(self, inputs, indices, labels=None, outputs=None,
                 add_input=False):
        """ Initialize the class.

        Parameters
        ----------
        inputs: numpy array
            the input data.
        indices: iterable of int
            the list of indices that is considered in this dataset.
        outputs: numpy array
            the output data.
        add_input: bool, default False
            if set concatenate the input data to the output (useful with
            auto-encoder).
        """
        if labels is not None:
            assert len(inputs) == len(labels)
        if outputs is not None:
            assert len(inputs) == len(outputs)
        self.inputs = inputs
        self.labels = labels
        self.outputs = outputs
        self.indices = indices
        self.add_input = add_input
        if self.add_input and self.outputs is None:
            self.outputs = self.inputs
            self.add_input = False

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: namedtuple
            a named tuple containing 'inputs', 'outputs', and 'labels' data.
        """
        if isinstance(item, int):
            concat_axis = 0
        else:
            concat_axis = 1
        _inputs = self.inputs[self.indices[item]]
        _labels, _outputs = (None, None)
        if self.labels is not None:
            _labels = self.labels[self.indices[item]]
        if self.outputs is not None:
            if self.add_input:
                _outputs = np.concatenate(
                    (self.outputs[self.indices[item]], _inputs),
                    axis=concat_axis)
            else:
                _outputs = self.outputs[self.indices[item]]
        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)

    def __len__(self):
        """ Return the length of the dataset.
        """
        return len(self.indices)

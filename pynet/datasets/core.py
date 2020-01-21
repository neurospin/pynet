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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
                 labels=None, stratify_label=None, custom_stratification=None,
                 projection_labels=None, number_of_folds=10, batch_size=1, weighted_random_sampler=False,
                 input_transforms=None, output_transforms=None, labels_transforms=None, stratify_label_transforms=None,
                 data_augmentation=None, add_input=False, patch_size=None, input_size=None, test_size=0.1,
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
        custom_stratification: dict, default None
            same format as projection labels. It will split the dataset into train/test/val according
            to the stratification defined in the dict.
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.
        number_of_folds: int, default 10
            the number of folds that will be used in the cross validation.
        batch_size: int, default 1
            the size of each mini-batch.
        weighted_random_sampler: bool, default False
            Whether we use a weighted random sampler (to deal with imbalanced classes issue) or not
        input_transforms, output_transforms: list of callable, default None
            transforms a list of samples with pre-defined transformations.
        data_augmentation: list of callable, default None
            transforms the training dataset input with pre-defined transformations on the fly during the training.
        add_input: bool, default False
            if true concatenate the input tensor to the output tensor.
        test_size: float, default 0.1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split.
        """
        df = pd.read_csv(metadata_path, sep="\t")
        mask = DataManager.get_mask(
            df=df,
            projection_labels=projection_labels)

        mask_indices = DataManager.get_indices_from_mask(mask)

        # We should only work with masked data but we want to preserve the memory mapping so we are getting the right
        # index at the end (in __getitem__ of ArrayDataset)

        self.inputs = np.load(input_path, mmap_mode='r')
        self.outputs, self.labels, self.stratify_label = (None, None, None)
        if output_path is not None:
            self.outputs = np.load(output_path, mmap_mode='r')

        if labels is not None:
            self.labels = df[labels].values
            self.labels = self.labels.squeeze()

        if stratify_label is not None:
            self.stratify_label = df[stratify_label].values
            # Apply the labels transform here as a mapping to the integer representation of the classes
            for i in mask_indices:
                label = self.stratify_label[i]
                for tf in (stratify_label_transforms or []):
                    label = tf(label)
                self.stratify_label[i] = label

        self.number_of_folds = number_of_folds
        self.batch_size = batch_size
        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []
        self.labels_transforms = labels_transforms or []
        self.data_augmentation = data_augmentation or []
        self.add_input = add_input
        self.data_loader_kwargs = dataloader_kwargs
        self.weighted_random_sampler = weighted_random_sampler

        if self.weighted_random_sampler:
            if self.stratify_label is None:
                raise ValueError('Impossible to use the WeightedRandomSampler if no stratify label is available.')
            class_samples_count = [0 for _ in range(len(set(self.stratify_label[mask])))] # len == nb of classes
            for label in self.stratify_label[mask]:
                class_samples_count[label] += 1
            # Imbalanced weights in case of imbalanced classes issue
            self.sampler_weigths = 1. / torch.tensor(class_samples_count, dtype=torch.float)

        self.dataset = dict((key, [])
                            for key in ("train", "test", "validation"))

        # 1st step: split into train/test (get only indices)
        dummy_like_X_masked = np.ones(np.sum(mask))
        val_indices, train_indices, test_indices = (None, None, None)
        if custom_stratification is not None:
            if "validation" in custom_stratification and stratify_label is not None:
                print("Warning: impossible to stratify the data: validation+test set already defined ! ")
            train_mask, test_mask = (DataManager.get_mask(df, custom_stratification["train"]),
                                     DataManager.get_mask(df, custom_stratification["test"]))
            if "validation" in custom_stratification:
                val_mask = DataManager.get_mask(df, custom_stratification["validation"])
                val_mask &= mask
                val_indices = DataManager.get_indices_from_mask(val_indices)
                self.number_of_folds = 1

            train_mask &= mask
            test_mask &= mask
            train_indices = DataManager.get_indices_from_mask(train_mask)
            test_indices = DataManager.get_indices_from_mask(test_mask)

        elif stratify_label is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=1, random_state=0, test_size=test_size)
            train_indices, test_indices = next(
                splitter.split(dummy_like_X_masked, self.stratify_label[mask]))
            train_indices = mask_indices[train_indices]
            test_indices = mask_indices[test_indices]
        else:
            if test_size == 1:
                train_indices, test_indices = (None, mask_indices)
            else:
                splitter = ShuffleSplit(
                    n_splits=1, random_state=0, test_size=test_size)
                train_indices, test_indices = next(splitter.split(dummy_like_X_masked))
                train_indices = mask_indices[train_indices]
                test_indices = mask_indices[test_indices]
        if self.labels is not None:
            assert len(self.labels) == len(self.inputs)
        self.dataset["test"] = ArrayDataset(
            self.inputs, test_indices, labels=self.labels,
            outputs=self.outputs, add_input=self.add_input,
            input_transforms = self.input_transforms,
            output_transforms = self.output_transforms,
            label_transforms = self.labels_transforms,
            patch_size=patch_size, input_size=input_size)

        if train_indices is None:
            return

        # 2nd step: split the training set into K folds (K-1 for training, 1
        # for validation, K times)
        dummy_like_X_train = np.ones(len(train_indices))
        if val_indices is not None:
            gen = [(train_indices, val_indices)]

        elif stratify_label is not None:
            kfold_splitter = StratifiedKFold(
                n_splits=number_of_folds)
            gen = kfold_splitter.split(
                dummy_like_X_train,
                np.array(self.stratify_label[train_indices], dtype=np.int32))
            gen = [(train_indices[tr], train_indices[val]) for (tr, val) in gen]

        else:
            kfold_splitter = KFold(n_splits=number_of_folds)
            gen = kfold_splitter.split(dummy_like_X_train)
            gen = [(train_indices[tr], train_indices[val]) for (tr, val) in gen]

        for fold_train_index, fold_val_index in gen:
            assert len(set(fold_val_index) & set(fold_train_index)) == 0
            assert len(set(fold_val_index)) + len(set(fold_train_index)) == len(set(train_indices))
            train_dataset = ArrayDataset(
                self.inputs, fold_train_index,
                labels=self.labels, outputs=self.outputs,
                add_input=self.add_input, input_transforms=self.input_transforms+self.data_augmentation,
                output_transforms=self.output_transforms+self.data_augmentation,
                label_transforms=self.labels_transforms,
                patch_size=patch_size, input_size=input_size)
            val_dataset = ArrayDataset(
                self.inputs, fold_val_index,
                labels=self.labels, outputs=self.outputs,
                add_input=self.add_input, input_transforms=self.input_transforms,
                output_transforms=self.output_transforms,
                label_transforms=self.labels_transforms,
                patch_size=patch_size, input_size=input_size
            )
            self.dataset["train"].append(train_dataset)
            self.dataset["validation"].append(val_dataset)


    @staticmethod
    def get_indices_from_mask(mask):
        return np.arange(len(mask))[mask]

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
        for key in ("inputs", "outputs", "labels"):
            if len(list_samples) == 0 or getattr(list_samples[-1], key) is None:
                data[key] = None
            else:
                data[key] = torch.stack([torch.as_tensor(getattr(s, key)) for s in list_samples], dim=0).float()
        if data["labels"] is not None:
            data["labels"] = data["labels"].type(torch.FloatTensor)
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
        _test, _train, _validation, sampler = (None, None, None, None)
        if test:
            _test = DataLoader(
                self.dataset["test"], batch_size=self.batch_size,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if train:
            if self.weighted_random_sampler:
                indices = self.dataset["train"][fold_index].indices
                samples_weigths = self.sampler_weigths[np.array(self.stratify_label[indices], dtype=np.int32)]
                sampler = WeightedRandomSampler(samples_weigths, len(indices), replacement=True)
            _train = DataLoader(
                self.dataset["train"][fold_index], batch_size=self.batch_size, sampler=sampler,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if validation:
            _validation = DataLoader(
                self.dataset["validation"][fold_index],
                batch_size=self.batch_size, collate_fn=self.collate_fn,
                **self.data_loader_kwargs)
        return SetItem(test=_test, train=_train, validation=_validation)

    @staticmethod
    def get_mask(df, projection_labels=None):
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
        mask: a list of boolean values
        """

        mask = np.ones(len(df), dtype=np.bool)
        if projection_labels is None:
            return mask
        for (col, val) in projection_labels.items():
            if isinstance(val, list):
                mask &= getattr(df, col).isin(val)
            elif val is not None:
                mask &= getattr(df, col).eq(val)
        return mask


class ArrayDataset(Dataset):
    """ A dataset based on numpy array.
    """
    def __init__(self, inputs, indices, labels=None, outputs=None,
                 add_input=False, input_transforms=None, output_transforms=None,
                 label_transforms=None, patch_size=None, input_size=None):
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
        patch_size: tuple, default None
            if set, return only patches of the input image
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
        self.patch_size = patch_size
        self.input_size = input_size
        if self.patch_size is not None:
            assert np.array(self.patch_size).shape == np.array(self.input_size).shape
            self.nb_patches_by_img = np.product(np.array(self.input_size) // np.array(self.patch_size))
            self.input_cached, self.output_cached, self.label_cached, self.input_indx_cached = None, None, None, None

        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []
        self.labels_transforms = label_transforms or []
        self.output_same_as_input = (self.add_input and self.outputs is None)
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

        if self.patch_size is not None:
            offset = item % self.nb_patches_by_img
            item = item // self.nb_patches_by_img
            if self.input_indx_cached == item:
                # Retrieve directly the input (and eventually the output)
                indx = np.unravel_index(offset, np.array(self.input_size) // np.array(self.patch_size))
                _inputs = self.input_cached[indx]
                _outputs, _labels = self.output_cached, self.label_cached
                if self.output_same_as_input:
                    _outputs = self.output_cached[indx]
                return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)

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

        # Apply the transformations to the data
        for tf in self.input_transforms:
            _inputs = tf(_inputs)
        if _outputs is not None:
            for tf in self.output_transforms:
                _outputs = tf(_outputs)
        if _labels is not None:
            for tf in self.labels_transforms:
                _labels = tf(_labels)

        # Eventually, get only one patch of the input (and one patch of the corresponding output if add_input==True)
        if self.patch_size is not None:
            self.input_indx_cached = item
            from skimage.util.shape import view_as_blocks
            # from a flat index, convert it to an nd-array index
            indx = np.unravel_index(offset, np.array(self.input_size) // np.array(self.patch_size))

            # Store everything in a cache to avoid useless computations...
            self.input_cached = view_as_blocks(_inputs, tuple(self.patch_size))
            self.output_cached = _outputs
            if self.output_same_as_input:
                self.output_cached = view_as_blocks(_outputs, tuple(self.patch_size))
            self.label_cached = _labels

            _inputs = self.input_cached[indx]
            if self.output_same_as_input:
                _outputs = self.output_cached[indx]


        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)

    def __len__(self):
        """ Return the length of the dataset.
        """
        if self.patch_size is not None:
            return len(self.indices) * self.nb_patches_by_img
        return len(self.indices)

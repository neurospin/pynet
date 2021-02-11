# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides core functions to load and split a dataset.
"""

# Imports
from collections import namedtuple, OrderedDict, Counter
import progressbar
import inspect
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    Dataset, DataLoader, WeightedRandomSampler, RandomSampler,
    SequentialSampler, Sampler)
from sklearn.model_selection import (
    KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit)
from skimage.util.shape import view_as_blocks

# Global parameters
SetItem = namedtuple("SetItem", ["test", "train", "validation"])
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])
logger = logging.getLogger("pynet")


class DataManager(object):
    """ Data manager used to split a dataset in train, test and validation
    pytorch datasets.
    """
    def __init__(self, input_path, metadata_path, output_path=None,
                 labels=None, stratify_label=None, custom_stratification=None,
                 projection_labels=None, number_of_folds=10, batch_size=1,
                 sampler="random", input_transforms=None,
                 output_transforms=None, data_augmentation_transforms=None,
                 add_input=False, test_size=0.1, label_mapping=None,
                 patch_size=None, continuous_labels=False, sample_size=1,
                 **dataloader_kwargs):
        """ Splits an input numpy array using memory-mapping into three sets:
        test, train and validation. This function can stratify the data.

        The train/test indices are performed using a Stratified or not
        ShuffleSplit.

        TODO: In the case of custom stratification, enable the weighted random
        sampler.

        Parameters
        ----------
        input_path: str
            the path to the numpy array containing the input tensor data
            that will be splited/loaded or the dataset itself.
        metadata_path: str
            the path to the metadata table in tsv format.
        output_path: str, default None
            the path to the numpy array containing the output tensor data
            that will be splited/loaded.
        labels: list of str, default None
            in case of classification/regression, the name of the column(s)
            in the metadata table to be predicted.
        projection_labels: dict, default None
            selects only the data that match the conditions. Use this
            dictionary to filter the input data from the metadata table:
            {<column_name>: <value>}.
        stratify_label: str, default None
            the name of the column in the metadata table containing the label
            used during the stratification (mutuallty exclusive with
            'custom_stratification').
        custom_stratification: dict, default None
            split the dataset into train/validation/test according to the
            defined stratification strategy. The filtering is performed
            as for the labels projection (mutuallty exclusive with
            'stratify_label').
        number_of_folds: int, default 10
            the number of folds that will be used in the cross validation.
        batch_size: int, default 1
            the size of each mini-batch.
        sampler: str or Sampler, default 'random'
            whether we use a sequential, random or weighted random sampler
            (to deal with imbalanced classes issue) during the generation of
            the mini-batches: None, 'random', 'weighted_random' or a custom
            Sampler class.
        input_transforms, output_transforms: list of callable, default None
            transforms a list of samples with pre-defined transformations.
        data_augmentation_transforms: list of callable, default None
            transforms the training dataset input with pre-defined
            transformations on the fly during the training.
        add_input: bool, default False
            if true concatenate the input tensor to the output tensor.
        test_size: float, default 0.1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the test split.
        label_mapping: dict, default None
            a mapping that can be used to convert labels to be predicted
            (string to int conversion).
        patch_size: tuple, default None
            the size of the patches that will be extracted from the
            input/output images.
        continuous_labels: bool, default False
            if set consider labels as continuous values; ie. floats otherwise
            a discrete values, ie. integer.
        sample_size: float, default 1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset used by the manger (random selection that can be usefull
            during testing.
        """
        # Checks
        if stratify_label is not None and custom_stratification is not None:
            raise ValueError("You specified two stratification strategies.")
        if ((inspect.isclass(sampler) and not issubclass(sampler, Sampler)) and
                sampler not in (None, "random", "weighted_random")):
            raise ValueError("Unsupported sampler.")
        if sampler == "weighted_random" and stratify_label is None:
            raise ValueError(
                "Impossible to use the weighted sampler without a "
                "stratification label.")

        # Class parameters
        # We should only work with masked data but we want to preserve the
        # memory mapping so we are getting the right index at the end
        # (in __getitem__ of ArrayDataset)
        self.batch_size = batch_size
        self.number_of_folds = number_of_folds
        self.data_loader_kwargs = dataloader_kwargs
        self.sampler = sampler
        self.continuous_labels = continuous_labels
        self.multi_bloc = None
        if isinstance(input_path, dict):
            self.dataset = input_path
            return
        df = pd.read_csv(metadata_path, sep="\t")
        logger.debug("Metadata:\n{0}".format(df))
        mask = DataManager.get_mask(
            df=df, projection_labels=projection_labels,
            sample_size=sample_size)
        mask_indices = DataManager.get_mask_indices(mask)
        logger.debug("Projection labels: {0}".format(projection_labels))
        logger.debug("Mask: {0}".format(mask))
        logger.debug("Mask indices: {0}".format(mask_indices))
        self.inputs = np.load(input_path, mmap_mode='r')
        logger.debug("Inputs: {0}".format(self.inputs.shape))
        self.outputs, self.labels = (None, None)
        if output_path is not None:
            self.outputs = np.load(output_path, mmap_mode='r')
            logger.debug("Outputs: {0}".format(self.outputs.shape))
        if labels is not None:
            self.labels = df[labels].values.squeeze()
            logger.debug("Labels: {0}".format(self.labels.shape))
            assert len(self.labels) == len(self.inputs)
        self.metadata = df
        self.mask = mask
        self.test_size = test_size
        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []
        self.data_augmentation_transforms = data_augmentation_transforms or []
        self.add_input = add_input
        self.dataset = dict(
            (key, []) for key in ("train", "test", "validation"))

        # Split into train+validation/test: get only indices.
        val_indices, train_indices, test_indices = (None, None, None)
        (self.stratify_labels, self.stratify_categories,
         self.sampler_weights) = (None, None, None)
        if stratify_label is not None:
            self.stratify_labels = df[stratify_label].values
            self.stratify_categories = set(self.stratify_labels[mask])
            self.sampler_weights = Counter(self.stratify_labels[mask])
        if self.test_size == 0:
            train_indices = mask_indices
            test_indices = None
        else:
            dummy_mask_like = np.ones(np.sum(mask))
            if custom_stratification is not None:
                for key in ("train", "test"):
                    if key not in custom_stratification:
                        raise ValueError("Unformed custom straitification.")
                train_mask = DataManager.get_mask(
                    df, custom_stratification["train"])
                test_mask = DataManager.get_mask(
                    df, custom_stratification["test"])
                train_mask &= mask
                test_mask &= mask
                train_indices = DataManager.get_mask_indices(train_mask)
                test_indices = DataManager.get_mask_indices(test_mask)
                if "validation" in custom_stratification:
                    val_mask = DataManager.get_mask(
                        df, custom_stratification["validation"])
                    val_mask &= mask
                    val_indices = DataManager.get_mask_indices(val_indices)
            elif stratify_label is not None:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, random_state=0, test_size=self.test_size)
                train_mask, test_mask = next(splitter.split(
                    dummy_mask_like, self.stratify_labels[mask]))
                train_indices = mask_indices[train_mask]
                test_indices = mask_indices[test_mask]
            else:
                if test_size == 1:
                    train_indices, test_indices = (None, mask_indices)
                else:
                    splitter = ShuffleSplit(
                        n_splits=1, random_state=0, test_size=test_size)
                    train_indices, test_indices = next(splitter.split(
                        dummy_mask_like))
                    train_indices = mask_indices[train_indices]
                    test_indices = mask_indices[test_indices]
        logger.debug("Train+Validation indices: {0}-{1}".format(
            len(train_indices) if train_indices is not None else None,
            train_indices))
        logger.debug("Test indices: {0}-{1}".format(
            len(test_indices) if test_indices is not None else None,
            test_indices))
        if test_indices is None:
            self.dataset["test"] = None
        else:
            self.dataset["test"] = ArrayDataset(
                self.inputs, test_indices, labels=self.labels,
                outputs=self.outputs, add_input=self.add_input,
                input_transforms=self.input_transforms,
                output_transforms=self.output_transforms,
                label_mapping=label_mapping,
                patch_size=patch_size)
        if train_indices is None:
            return

        # Split the training set into K folds (K-1 for training, 1 for
        # validation, K times)
        dummy_train_like = np.ones(len(train_indices))
        if val_indices is not None:
            self.generator = [(train_indices, val_indices)]
        elif stratify_label is not None:
            kfold_splitter = StratifiedKFold(
                n_splits=self.number_of_folds)
            self.generator = kfold_splitter.split(
                dummy_train_like, self.stratify_labels[train_indices])
            self.generator = [(train_indices[train], train_indices[val])
                              for (train, val) in self.generator]
        else:
            kfold_splitter = KFold(n_splits=self.number_of_folds)
            self.generator = kfold_splitter.split(dummy_train_like)
            self.generator = [(train_indices[train], train_indices[val])
                              for (train, val) in self.generator]
        for fold_train_indices, fold_val_indices in self.generator:
            logger.debug("Fold train indices: {0}".format(fold_train_indices))
            logger.debug("Fold val indices: {0}".format(fold_val_indices))
            assert len(set(fold_val_indices) & set(fold_train_indices)) == 0
            assert (len(set(fold_val_indices)) + len(set(fold_train_indices))
                    == len(set(train_indices)))
            train_dataset = ArrayDataset(
                self.inputs, fold_train_indices, labels=self.labels,
                outputs=self.outputs, add_input=self.add_input,
                input_transforms=(self.input_transforms +
                                  self.data_augmentation_transforms),
                output_transforms=(self.output_transforms +
                                   self.data_augmentation_transforms),
                label_mapping=label_mapping,
                patch_size=patch_size)
            val_dataset = ArrayDataset(
                self.inputs, fold_val_indices, labels=self.labels,
                outputs=self.outputs, add_input=self.add_input,
                input_transforms=self.input_transforms,
                output_transforms=self.output_transforms,
                label_mapping=label_mapping,
                patch_size=patch_size)
            self.dataset["train"].append(train_dataset)
            self.dataset["validation"].append(val_dataset)

    @classmethod
    def from_numpy(cls, test_inputs=None, test_outputs=None, test_labels=None,
                   train_inputs=None, train_outputs=None, train_labels=None,
                   validation_inputs=None, validation_outputs=None,
                   validation_labels=None, batch_size=1, sampler="random",
                   input_transforms=None, output_transforms=None,
                   data_augmentation_transforms=None, add_input=False,
                   label_mapping=None, patch_size=None,
                   continuous_labels=False):
        """ Create a data manger from numpy arrays.

        Parameters
        ----------
        *_inputs, *_outputs, *_labels: ndarrays
            the training data.
        batch_size: int, default 1
            the size of each mini-batch.
        sampler: str or Sampler, default 'random'
            whether we use a sequential, random or weighted random sampler
            (to deal with imbalanced classes issue) during the generation of
            the mini-batches: None, 'random', 'weighted_random' or a custom
            Sampler class.
        input_transforms, output_transforms: list of callable, default None
            transforms a list of samples with pre-defined transformations.
        data_augmentation_transforms: list of callable, default None
            transforms the training dataset input with pre-defined
            transformations on the fly during the training.
        add_input: bool, default False
            if true concatenate the input tensor to the output tensor.
        label_mapping: dict, default None
            a mapping that can be used to convert labels to be predicted
            (string to int conversion).
        patch_size: tuple, default None
            the size of the patches that will be extracted from the
            input/output images.
        continuous_labels: bool, default False
            if set consider labels as continuous values; ie. floats otherwise
            a discrete values, ie. integer.

        Returns
        -------
        ins: DataManager
            a data manager.
        """
        dataset = dict((key, None) for key in ("train", "test", "validation"))
        input_transforms = input_transforms or []
        output_transforms = output_transforms or []
        data_augmentation_transforms = data_augmentation_transforms or []
        if test_inputs is not None:
            test_dataset = ArrayDataset(
                inputs=test_inputs, indices=range(len(test_inputs)),
                labels=test_labels, outputs=test_outputs,
                input_transforms=input_transforms,
                output_transforms=output_transforms,
                add_input=add_input,
                label_mapping=label_mapping,
                patch_size=patch_size)
            dataset["test"] = test_dataset
        if train_inputs is not None:
            train_dataset = ArrayDataset(
                inputs=train_inputs,
                indices=range(len(train_inputs)),
                labels=train_labels,
                outputs=train_outputs,
                input_transforms=(input_transforms +
                                  data_augmentation_transforms),
                output_transforms=(output_transforms +
                                   data_augmentation_transforms),
                add_input=add_input,
                label_mapping=label_mapping,
                patch_size=patch_size)
            dataset["train"] = [train_dataset]
        if validation_inputs is not None:
            validation_dataset = ArrayDataset(
                inputs=validation_inputs,
                indices=range(len(validation_inputs)),
                labels=validation_labels,
                outputs=validation_outputs,
                input_transforms=input_transforms,
                output_transforms=output_transforms,
                add_input=add_input,
                label_mapping=label_mapping,
                patch_size=patch_size)
            dataset["validation"] = [validation_dataset]
        return cls(input_path=dataset,
                   metadata_path=None,
                   sampler=sampler,
                   batch_size=batch_size,
                   number_of_folds=1,
                   continuous_labels=continuous_labels)

    @classmethod
    def from_dataset(cls, test_dataset=None, train_dataset=None,
                     validation_dataset=None, batch_size=1, sampler="random",
                     multi_bloc=False):
        """ Create a data manger from torch datasets.

        Parameters
        ----------
        *_dataset: Dataset
            the train/validation/test datasets.
        batch_size: int, default 1
            the size of each mini-batch.
        sampler: str or Sampler, default 'random'
            whether we use a sequential, random or weighted random sampler
            (to deal with imbalanced classes issue) during the generation of
            the mini-batches: None, 'random', 'weighted_random' or a custom
            Sampler class.
        multi_bloc: bool, default False
            if sett expect multi bloc datasets that returns a list with
            N bloc of data.

        Returns
        -------
        ins: DataManager
            a data manager.
        """
        dataset = dict((key, None) for key in ("train", "test", "validation"))
        input_transforms = []
        output_transforms = []
        data_augmentation_transforms = []
        if test_dataset is not None:
            dataset["test"] = test_dataset
        if train_dataset is not None:
            dataset["train"] = [train_dataset]
        if validation_dataset is not None:
            dataset["validation"] = [validation_dataset]
        manager = cls(input_path=dataset,
                      metadata_path=None,
                      sampler=sampler,
                      batch_size=batch_size,
                      number_of_folds=1)
        manager.multi_bloc = multi_bloc
        return manager

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: Dataset or list of Dataset
            the requested set of data: test, train or validation.
        """
        if item not in ("train", "test", "validation"):
            raise ValueError(
                "Unknown set! Must be 'train', 'test' or 'validation'.")
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
            if (len(list_samples) == 0 or
                    getattr(list_samples[-1], key) is None):
                data[key] = None
            elif self.multi_bloc:
                n_blocs = len(getattr(list_samples[-1], key))
                data[key] = [torch.stack([
                    torch.as_tensor(getattr(sample, key)[bloc])
                    for sample in list_samples], dim=0).float()
                    for bloc in range(n_blocs)]
            else:
                data[key] = torch.stack([
                    torch.as_tensor(getattr(sample, key))
                    for sample in list_samples], dim=0).float()
        if data["labels"] is not None:
            if self.continuous_labels:
                data["labels"] = data["labels"].type(torch.FloatTensor)
            else:
                data["labels"] = data["labels"].type(torch.LongTensor)
        return DataItem(**data)

    def get_dataloader(self, train=False, validation=False, test=False,
                       fold_index=0):
        """ Generate a pytorch DataLoader.

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
            # weights is a list of weights per data point in the data set we
            # are drawing from, NOT a weight per class.
            if inspect.isclass(self.sampler):
                sampler = self.sampler(self.dataset["train"][fold_index])
            elif self.sampler == "weighted_random":
                if self.sampler_weights is None:
                    raise ValueError(
                        "Weighted random not yet supported with your input "
                        "parameters.")
                indices = self.dataset["train"][fold_index].indices
                samples_weigths = [self.sampler_weights[
                    self.stratify_labels[idx]] for idx in indices]
                sampler = WeightedRandomSampler(
                    samples_weigths, len(indices), replacement=True)
            elif self.sampler == "random":
                sampler = RandomSampler(
                    self.dataset["train"][fold_index], replacement=False)
            _train = DataLoader(
                self.dataset["train"][fold_index], batch_size=self.batch_size,
                sampler=sampler, collate_fn=self.collate_fn,
                **self.data_loader_kwargs)
        if validation:
            _validation = DataLoader(
                self.dataset["validation"][fold_index],
                batch_size=self.batch_size, collate_fn=self.collate_fn,
                **self.data_loader_kwargs)
        return SetItem(test=_test, train=_train, validation=_validation)

    @staticmethod
    def get_mask(df, projection_labels=None, sample_size=1):
        """ Filter a table.

        Parameters
        ----------
        df: a pandas DataFrame
            a table data.
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.
        sample_size: float, default 1
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset used by the manager (random selection that can be usefull
            during testing).

        Returns
        -------
        mask: a list of boolean values.
        """
        mask = np.random.choice(2, len(df), p=[1 - sample_size, sample_size])
        mask = mask.astype(np.bool)
        if projection_labels is None:
            return mask
        for (col, val) in projection_labels.items():
            if isinstance(val, list):
                mask &= getattr(df, col).isin(val)
            elif val is not None:
                mask &= getattr(df, col).eq(val)
        return mask

    @staticmethod
    def get_mask_indices(mask):
        """ From an input mask vector, return the true indices.
        """
        return np.arange(len(mask))[mask]


class ArrayDataset(Dataset):
    """ A dataset based on numpy array.
    """
    def __init__(self, inputs, indices, labels=None, outputs=None,
                 add_input=False, input_transforms=None,
                 output_transforms=None, label_mapping=None,
                 patch_size=None):
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
        input_transforms, output_transforms: list of callable, default None
            transforms a list of samples with pre-defined transformations.
        label_mapping: dict, default None
            a mapping that can be used to convert labels to be predicted
            (string to int conversion).
        patch_size: tuple, default None
            the size of the patches that will be extracted from the
            input/output images.
        """
        # Checks
        if labels is not None:
            assert len(inputs) == len(labels)
        if outputs is not None:
            assert len(inputs) == len(outputs)

        # Class parameters
        self.inputs = inputs
        self.labels = labels
        self.outputs = outputs
        self.indices = indices
        self.add_input = add_input
        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []
        self.label_mapping = label_mapping
        self.patch_size = patch_size
        self.input_size = np.asarray(self.inputs.shape[2:])
        if self.patch_size is not None:
            self.patch_size = np.asarray(self.patch_size)
            logger.debug("Patch size: {0}".format(self.patch_size))
            logger.debug("Input size: {0}".format(self.input_size))
            assert self.patch_size.shape == self.input_size.shape
            self.patch_grid = self.input_size // self.patch_size
            logger.debug("Patch grid: {0}".format(self.patch_grid))
            self.nb_patches_by_img = np.prod(self.patch_grid)
            logger.debug("Number patches: {0}".format(self.nb_patches_by_img))
            (self.input_cached, self.output_cached, self.label_cached,
             self.image_idx_cached) = (None, None, None, None)

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: namedtuple
            a named tuple containing 'inputs', 'outputs', and 'labels' data.
        """
        logger.debug("Asked item: {0}".format(item))
        if isinstance(item, int):
            concat_axis = 0
        else:
            concat_axis = 1

        # If the patches are already loaded just select the requested patch
        if self.patch_size is not None:
            patch_idx = item % self.nb_patches_by_img
            image_idx = item // self.nb_patches_by_img
            indices = self.indices[image_idx]
            if self.image_idx_cached == image_idx:
                # Retrieve directly the input (and eventually the output)
                idx = tuple(np.unravel_index(patch_idx, self.patch_grid))
                logger.debug("Getting patch index item: {0}".format(idx))
                _inputs = self.input_cached[idx]
                if self.output_cached is not None:
                    _outputs = self.output_cached[idx]
                else:
                    _outputs = None
                _labels = self.label_cached
                return DataItem(inputs=_inputs, outputs=_outputs,
                                labels=_labels)
        else:
            indices = self.indices[item]

        # Load the requested data
        logger.debug("Precomputed indices: {0}".format(indices))
        _inputs = self.inputs[indices]
        _labels, _outputs = (None, None)
        if self.labels is not None:
            _labels = self.labels[indices]
        if self.outputs is not None:
            _outputs = self.outputs[indices]

        # Apply the transformations to the data
        seed = random.getrandbits(30)
        for tf in self.input_transforms:
            if hasattr(tf, "seed"):
                tf.seed = seed
            if hasattr(tf, "dtype"):
                tf.dtype = "input"
            _inputs = tf(_inputs)
        if _outputs is not None:
            for tf in self.output_transforms:
                if hasattr(tf, "seed"):
                    tf.seed = seed
                if hasattr(tf, "dtype"):
                    tf.dtype = "output"
                _outputs = tf(_outputs)
        if _labels is not None and self.label_mapping is not None:
            _labels = [label_mapping[item] for item in _labels]

        # Cache data patches and select the requested patch
        if self.patch_size is not None:
            self.image_idx_cached = image_idx
            idx = tuple(np.unravel_index(patch_idx, self.patch_grid))
            logger.debug("Getting patch index item: {0}".format(idx))
            logger.debug("Splitting input: {0}".format(_inputs.shape))
            self.input_cached = ArrayDataset._create_patches(
                _inputs, self.patch_size)
            logger.debug("Cached: {0}".format(self.input_cached.shape))
            if _outputs is not None:
                logger.debug("Splitting output: {0}".format(_outputs.shape))
                self.output_cached = ArrayDataset._create_patches(
                    _outputs, self.patch_size)
                logger.debug("Cached: {0}".format(self.output_cached.shape))
            self.label_cached = _labels
            _inputs = self.input_cached[idx]
            _outputs = self.output_cached[idx]

        # Add input
        if self.add_input:
            if _outputs is None:
                _outputs = _inputs
            else:
                _outputs = np.concatenate(
                    (_outputs, _inputs), axis=concat_axis)

        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels)

    @staticmethod
    def _create_patches(arr, patch_size):
        channel_idx = len(patch_size)
        channels_cached = []
        for channel in arr:
            channel_patches = view_as_blocks(channel, tuple(patch_size))
            channel_patches = np.expand_dims(channel_patches, axis=channel_idx)
            channels_cached.append(channel_patches)
        return np.concatenate(channels_cached, axis=channel_idx)

    def __len__(self):
        """ Return the length of the dataset.
        """
        if self.patch_size is not None:
            return len(self.indices) * self.nb_patches_by_img
        return len(self.indices)

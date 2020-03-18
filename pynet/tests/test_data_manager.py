# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import copy
import sys
import numpy as np
import pandas as pd
import unittest.mock as mock
from unittest.mock import patch


# Package import
from pynet.datasets.core import DataManager, ArrayDataset


class TestDataManager(unittest.TestCase):
    """ Test the DataManager class.
    """
    def setUp(self):
        """ Setup test.
        """
        self.input_arr = np.ones((10, 1, 4, 4))
        for item in range(self.input_arr.shape[0]):
            self.input_arr[item] *= item
        self.output_arr = np.ones((10, 2, 4, 4))
        offset = self.input_arr.shape[0]
        for item in range(offset, self.output_arr.shape[0] + offset):
            self.output_arr[item - offset] *= item
        data = {
            "label": ["group1"] * 5 + ["group2"] * 5,
            "age": np.random.randint(20, 60, 10),
            "sex": ["M"] * 6 + ["F"] * 4
        }
        self.metadata = pd.DataFrame.from_dict(data)
        self.kwargs = {
            "input_path": "/my/path/mock_input_path",
            "metadata_path": "/my/path/mock_metadata_path",
            "output_path": "/my/path/mock_output_path",
            "labels": None,
            "stratify_label": None,
            "custom_stratification": None,
            "projection_labels": None,
            "number_of_folds": 3,
            "batch_size": 2,
            "sampler": None,
            "input_transforms": None,
            "output_transforms": None,
            "data_augmentation_transforms": None,
            "add_input": False,
            "test_size": 0.1,
            "label_mapping": None,
            "patch_size": None
        }

    def tearDown(self):
        """ Run after each test.
        """
        pass

    @mock.patch("pandas.read_csv")
    @mock.patch("numpy.load")
    def test_filtering(self, mock_load, mock_readcsv):
        """ Test the filtering behaviour.
        """
        # Set the mocked function returned values.
        mock_readcsv.return_value = self.metadata
        mock_load.side_effect = [self.input_arr, self.output_arr]
        kwargs = copy.deepcopy(self.kwargs)
        kwargs["projection_labels"] = {
            "sex": "M"
        }
        kwargs["test_size"] = 0.2
        kwargs["number_of_folds"] = 2
        batch_size = kwargs["batch_size"]

        # Test execution
        manager = DataManager(**kwargs)
        self.assertTrue(all(
            manager.metadata[manager.mask]["sex"].values == "M"))

    @mock.patch("pandas.read_csv")
    @mock.patch("numpy.load")
    def test_default_sampler(self, mock_load, mock_readcsv):
        """ Test the default sampler behaviour.
        """
        # Set the mocked function returned values.
        mock_readcsv.return_value = self.metadata
        mock_load.side_effect = [
            self.input_arr, self.output_arr, self.input_arr, self.output_arr]
        kwargs = copy.deepcopy(self.kwargs)
        batch_size = kwargs["batch_size"]

        # Test execution
        manager = DataManager(**kwargs)
        self.assertEqual(
            manager.number_of_folds, kwargs["number_of_folds"])
        for fold in range(manager.number_of_folds):
            loaders = manager.get_dataloader(
                train=True,
                validation=True,
                fold_index=fold)
            fold_train_indices, fold_val_indices = manager.generator[fold]
            train_div, train_rest = divmod(len(fold_train_indices), batch_size)
            val_div, val_rest = divmod(len(fold_val_indices), batch_size)
            for iteration, dataitem in enumerate(loaders.train):
                for key, refarr in (("inputs", self.input_arr),
                                    ("outputs", self.output_arr)):
                    arr = getattr(dataitem, key)
                    refshape = list(refarr.shape)
                    start = batch_size * iteration
                    if iteration == train_div:
                        refshape[0] = train_rest
                        stop = batch_size * iteration + train_rest
                    else:
                        refshape[0] = batch_size
                        stop = batch_size * (iteration + 1)
                    indices = fold_train_indices[start: stop]
                    self.assertTrue(np.allclose(refarr[indices], arr))
                    self.assertEqual(list(arr.shape), refshape)
            for iteration, dataitem in enumerate(loaders.validation):
                for key, refarr in (("inputs", self.input_arr),
                                    ("outputs", self.output_arr)):
                    arr = getattr(dataitem, key)
                    refshape = list(refarr.shape)
                    start = batch_size * iteration
                    if iteration == val_div:
                        refshape[0] = val_rest
                        stop = batch_size * iteration + val_rest
                    else:
                        refshape[0] = batch_size
                        stop = batch_size * (iteration + 1)
                    indices = fold_val_indices[start: stop]
                    self.assertTrue(np.allclose(refarr[indices], arr))
                    self.assertEqual(list(arr.shape), refshape)

        # Test execution with the add input option
        kwargs["add_input"] = True
        manager = DataManager(**kwargs)
        loaders = manager.get_dataloader(
            train=True,
            fold_index=0)
        expected_nb_channels = (
            self.output_arr.shape[1] + self.input_arr.shape[1])
        for iteration, dataitem in enumerate(loaders.train):
            self.assertEqual(dataitem.outputs.shape[1], expected_nb_channels)

    @mock.patch("pandas.read_csv")
    @mock.patch("numpy.load")
    def test_random_sampler(self, mock_load, mock_readcsv):
        """ Test the random sampler behaviour.
        """
        # Set the mocked function returned values.
        mock_readcsv.return_value = self.metadata
        mock_load.side_effect = [self.input_arr, self.output_arr]
        kwargs = copy.deepcopy(self.kwargs)
        batch_size = kwargs["batch_size"]
        kwargs["sampler"] = "random"

        # Test execution
        manager = DataManager(**kwargs)
        self.assertEqual(
            manager.number_of_folds, kwargs["number_of_folds"])
        for fold in range(manager.number_of_folds):
            loaders = manager.get_dataloader(
                train=True,
                validation=True,
                fold_index=fold)
            fold_train_indices, fold_val_indices = manager.generator[fold]
            train_div, train_rest = divmod(len(fold_train_indices), batch_size)
            val_div, val_rest = divmod(len(fold_val_indices), batch_size)
            for iteration, dataitem in enumerate(loaders.train):
                for key, refarr in (("inputs", self.input_arr),
                                    ("outputs", self.output_arr)):
                    arr = getattr(dataitem, key)
                    refshape = list(refarr.shape)
                    if iteration == train_div:
                        refshape[0] = train_rest
                    else:
                        refshape[0] = batch_size
                    self.assertEqual(list(arr.shape), refshape)
            for iteration, dataitem in enumerate(loaders.validation):
                for key, refarr in (("inputs", self.input_arr),
                                    ("outputs", self.output_arr)):
                    arr = getattr(dataitem, key)
                    refshape = list(refarr.shape)
                    if iteration == val_div:
                        refshape[0] = val_rest
                    else:
                        refshape[0] = batch_size
                    self.assertEqual(list(arr.shape), refshape)

    @mock.patch("pandas.read_csv")
    @mock.patch("numpy.load")
    def test_weighted_random_sampler(self, mock_load, mock_readcsv):
        """ Test the weighted radom sampler behaviour.
        """
        # Set the mocked function returned values.
        mock_readcsv.return_value = self.metadata
        mock_load.side_effect = [self.input_arr, self.output_arr]
        kwargs = copy.deepcopy(self.kwargs)
        batch_size = kwargs["batch_size"]
        kwargs["test_size"] = 0.2
        kwargs["stratify_label"] = "label"
        kwargs["sampler"] = "weighted_random"

        # Test execution
        manager = DataManager(**kwargs)
        self.assertEqual(
            manager.number_of_folds, kwargs["number_of_folds"])
        for fold in range(manager.number_of_folds):
            loaders = manager.get_dataloader(
                train=True,
                validation=True,
                fold_index=fold)
            fold_train_indices, fold_val_indices = manager.generator[fold]
            train_div, train_rest = divmod(len(fold_train_indices), batch_size)
            val_div, val_rest = divmod(len(fold_val_indices), batch_size)
            for iteration, dataitem in enumerate(loaders.train):
                for key, refarr in (("inputs", self.input_arr),
                                    ("outputs", self.output_arr)):
                    arr = getattr(dataitem, key)
                    refshape = list(refarr.shape)
                    if iteration == train_div:
                        refshape[0] = train_rest
                    else:
                        refshape[0] = batch_size
                    self.assertEqual(list(arr.shape), refshape)
            for iteration, dataitem in enumerate(loaders.validation):
                for key, refarr in (("inputs", self.input_arr),
                                    ("outputs", self.output_arr)):
                    arr = getattr(dataitem, key)
                    refshape = list(refarr.shape)
                    if iteration == val_div:
                        refshape[0] = val_rest
                    else:
                        refshape[0] = batch_size
                    self.assertEqual(list(arr.shape), refshape)

    @mock.patch("pandas.read_csv")
    @mock.patch("numpy.load")
    def test_patches(self, mock_load, mock_readcsv):
        """ Test the patch accessor behaviour.
        """
        # Set the mocked function returned values.
        mock_readcsv.return_value = self.metadata
        mock_load.side_effect = [self.input_arr, self.output_arr]
        kwargs = copy.deepcopy(self.kwargs)
        batch_size = kwargs["batch_size"]
        kwargs["patch_size"] = (2, 2)

        # Test execution
        manager = DataManager(**kwargs)
        self.assertEqual(
            manager.number_of_folds, kwargs["number_of_folds"])
        for fold in range(manager.number_of_folds):
            loaders = manager.get_dataloader(
                train=True,
                fold_index=fold)
            fold_train_indices, fold_val_indices = manager.generator[fold]
            train_div, train_rest = divmod(len(fold_train_indices), batch_size)
            val_div, val_rest = divmod(len(fold_val_indices), batch_size)
            for iteration, dataitem in enumerate(loaders.train):
                for key, refarr in (("inputs", self.input_arr),
                                    ("outputs", self.output_arr)):
                    arr = getattr(dataitem, key)
                    self.assertTrue(np.allclose(
                        arr.shape[-2:], kwargs["patch_size"]))


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()

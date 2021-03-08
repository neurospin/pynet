# -*- coding: utf-8 -*-
########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
########################################################################

"""
Module provides functions to prepare different datasets from EUAIMS.
"""

# Imports
import os
import json
import time
import urllib
import shutil
import pickle
import requests
import logging
import numpy as np
from collections import namedtuple
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from pynet.datasets import Fetchers
from neurocombat_sklearn import CombatModel as fortin_combat
from nibabel.freesurfer.mghformat import load as surface_loader


# Global parameters
Item = namedtuple("Item", ["train_input_path", "test_input_path",
                           "train_metadata_path", "test_metadata_path"])
COHORT_NAME = "EUAIMS"
FOLDER = "/neurospin/brainomics/2020_deepint/data"
SAVING_FOLDER = "/tmp/EUAIMS"
FILES = {
    "stratification": os.path.join(FOLDER, "EUAIMS_stratification.tsv"),
    "rois_mapper": os.path.join(FOLDER, "EUAIMS_rois.tsv"),
    "surf_stratification": os.path.join(
        FOLDER, "EUAIMS_surf_stratification.tsv")
}

DEFAULTS = {
    "clinical": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True,
        "drop_cols": ["t1:site", "t1:ageyrs", "t1:sex", "t1:fsiq",
                      "t1:group", "t1:diagnosis", "mri", "t1:group:name",
                      "qc", "labels", "subgroups"],
        "qc": {"t1:fsiq": {"gte": 70},
               "mri": {"eq": 1},
               "qc": {"eq": "include"}}
    },
    "rois": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True, "adjust_sites": True,
        "metrics": ["lgi:avg", "thick:avg", "surf:area"],
        "roi_types": ["cortical"],
        "residualize_by": {"continuous": ["t1:ageyrs", "t1:fsiq"],
                           "discrete": ["t1:sex"]},
        "qc": {"t1:fsiq": {"gte": 70},
               "mri": {"eq": 1},
               "qc": {"eq": "include"}}
    },
    "genetic": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True, "scores": None,
        "qc": {"t1:fsiq": {"gte": 70},
               "mri": {"eq": 1},
               "qc": {"eq": "include"}}
    },
    "surface": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True, "adjust_sites": True,
        "metrics": ["pial_lgi", "thickness"],
        "residualize_by": {"continuous": ["t1:ageyrs", "t1:fsiq"],
                           "discrete": ["t1:sex"]},
        "qc": {"t1:fsiq": {"gte": 70},
               "mri": {"eq": 1},
               "qc": {"eq": "include"}}
    },
    "multiblock": {
        "test_size": 0.2, "seed": 42,
        "blocks": ["clinical", "surface-lh", "surface-rh", "genetic"],
        "qc": {"t1:fsiq": {"gte": 70},
               "mri": {"eq": 1},
               "qc": {"eq": "include"}}
    }
}
logger = logging.getLogger("pynet")


def apply_qc(data, prefix, qc):
    """ applies quality control to the data

    Parameters
    ----------
    data: pandas DataFrame
        data for which we control the quality
    prefix: string
        prefix of the column names
    qc: dict
        quality control dict. keys are the name of the columns
        to control on, and values dict containing an order relationsip
        and a value as items

    Returns
    -------
    data: pandas DataFrame
        selected data by the quality control
    """
    idx_to_keep = pd.Series([True] * len(data))

    relation_mapper = {
        "gt": lambda x, y: x > y,
        "lt": lambda x, y: x < y,
        "gte": lambda x, y: x >= y,
        "lte": lambda x, y: x <= y,
        "eq": lambda x, y: x == y,
    }
    for name, controls in qc.items():
        for relation, value in controls.items():
            if relation not in relation_mapper.keys():
                raise ValueError("The relationship {} provided is not a \
                    valid one".format(relation))
            elif "{}{}".format(prefix, name) in data.columns:
                new_idx = relation_mapper[relation](
                    data["{}{}".format(prefix, name)], value)
                idx_to_keep = idx_to_keep & new_idx
    return data[idx_to_keep]


def fetch_clinical_wrapper(datasetdir=SAVING_FOLDER, files=FILES,
                           cohort=COHORT_NAME, defaults=DEFAULTS['clinical']):
    """ Fetcher wrapper for clinical data

    Parameters
    ----------
    datasetdir: string, default SAVING_FOLDER
        path to the folder in which to save the data
    files: dict, default FILES
        contains the paths to the different files
    cohort: string, default COHORT_NAME,
        name of the cohort
    subject_columns_name: string, default 'subjects'
        name of the column containing the subjects id
    defaults: dict, default DEFAULTS
        default values for the wrapped function

    Returns
    -------
    fetcher: function
        corresponding fetcher.

    """

    fetcher_name = "fetcher_clinical_{}".format(cohort)

    # @Fetchers.register
    def fetch_clinical(
            test_size=defaults["test_size"], seed=defaults["seed"],
            return_data=defaults["return_data"], z_score=defaults["z_score"],
            drop_cols=defaults["drop_cols"], qc=defaults["qc"]):
        """ Fetches and preprocesses clinical data

        Parameters
        ----------
        test_size: float, default 0.2
            proportion of the dataset to keep for testing. Preprocessing models
            will only be fitted on the training part and applied to the test
            set. You can specify not to use a testing set by setting it to 0
        seed: int, default 42
            random seed to split the data into train / test
        return_data: bool, default False
            If false, saves the data in the specified folder, and return the
            path. Otherwise, returns the preprocessed data and the
            corresponding subjects
        z_score: bool, default True
            wether or not to transform the data into z_scores, meaning
            standardizing and scaling it
        drop_cols: list of string, see default
            names of the columns to drop before saving the data.
        qc: dict, see default
            keys are the name of the features the control on, values are the
            requirements on their values (see the function apply_qc)

        Returns
        -------
        item: namedtuple
            a named tuple containing 'train_input_path', 'train_metadata_path',
            and 'test_input_path', 'test_metadata_path' if test_size > 0
        X_train: numpy array,
            Training data, if return_data is True
        X_test: numpy array,
            Test data, if return_data is True and test_size > 0
        subj_train: numpy array,
            Training subjects, if return_data is True
        subj_test: numpy array,
            Test subjects, if return_data is True and test_size > 0
        """
        clinical_prefix = "bloc-clinical_score-"
        subject_column_name = "participant_id"
        path = os.path.join(datasetdir, "clinical_X_train.npy")
        meta_path = os.path.join(datasetdir, "clinical_X_train.tsv")
        path_test = None
        meta_path_test = None
        if test_size > 0:
            path_test = os.path.join(datasetdir, "clinical_X_test.npy")
            meta_path_test = os.path.join(datasetdir, "clinical_X_test.tsv")

        if not os.path.isfile(path):

            data = pd.read_csv(files["stratification"], sep="\t")
            clinical_cols = [subject_column_name]
            clinical_cols += [col for col in data.columns
                              if col.startswith(clinical_prefix)]
            data = data[clinical_cols]
            data_train = apply_qc(data, clinical_prefix, qc).sort_values(
                subject_column_name)
            data_train.columns = [elem.replace(clinical_prefix, "")
                                  for elem in data_train.columns]
            X_train = data_train.drop(columns=drop_cols)

            # Splits in train and test and removes nans
            X_test, subj_test = (None, None)
            if test_size > 0:
                X_train, X_test = train_test_split(
                    X_train, test_size=test_size, random_state=seed)
                na_idx_test = (X_test.isna().sum(1) == 0)
                X_test = X_test[na_idx_test]
                subj_test = X_test[subject_column_name].values
                X_test = X_test.drop(columns=[subject_column_name]).values
            na_idx_train = (X_train.isna().sum(1) == 0)
            X_train = X_train[na_idx_train]
            subj_train = X_train[subject_column_name].values
            X_train = X_train.drop(columns=[subject_column_name])
            cols = X_train.columns
            X_train = X_train.values

            # Standardizes and scales
            if z_score:
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                _path = os.path.join(datasetdir, "clinical_scaler.pkl")
                with open(_path, "wb") as f:
                    pickle.dump(scaler, f)
                if test_size > 0:
                    X_test = scaler.transform(X_test)

            # Return data and subjects

            X_train_df = pd.DataFrame(data=X_train, columns=cols)
            X_train_df.insert(0, subject_column_name, subj_train)
            X_test_df = None
            if test_size > 0:
                X_test_df = pd.DataFrame(data=X_test, columns=cols)
                X_test_df.insert(0, subject_column_name, subj_test)

            # Saving
            np.save(path, X_train)
            X_train_df.to_csv(meta_path, index=False, sep="\t")
            if test_size > 0:
                np.save(path_test, X_test)
                X_test_df.to_csv(meta_path_test, index=False, sep="\t")

        if return_data:
            X_train = np.load(path)
            subj_train = pd.read_csv(meta_path, sep="\t")[
                subject_column_name].values
            X_test, subj_test = (None, None)
            if test_size > 0:
                X_test = np.load(path_test)
                subj_test = pd.read_csv(meta_path_test, sep="\t")[
                    subject_column_name].values
            return X_train, X_test, subj_train, subj_test
        else:
            return Item(train_input_path=path, test_input_path=path_test,
                        train_metadata_path=meta_path,
                        test_metadata_path=meta_path_test)

    return fetch_clinical


def fetch_rois_wrapper(datasetdir=SAVING_FOLDER, files=FILES,
                       cohort=COHORT_NAME, site_column_name="t1:site",
                       defaults=DEFAULTS['rois']):
    """ Fetcher wrapper for rois data

    Parameters
    ----------
    datasetdir: string, default SAVING_FOLDER
        path to the folder in which to save the data
    files: dict, default FILES
        contains the paths to the different files
    cohort: string, default COHORT_NAME,
        name of the cohort
    site_columns_name: string, default "t1:site"
        name of the column containing the site of MRI acquisition
    defaults: dict, default DEFAULTS
        default values for the wrapped function

    Returns
    -------
    fetcher: function
        corresponding fetcher

    """

    fetcher_name = "fetcher_rois_{}".format(cohort)

    # @Fetchers.register
    def fetch_rois(
            metrics=defaults["metrics"], roi_types=defaults["roi_types"],
            test_size=defaults["test_size"], seed=defaults["seed"],
            return_data=defaults["return_data"], z_score=defaults["z_score"],
            adjust_sites=defaults["adjust_sites"],
            residualize_by=defaults["residualize_by"], qc=defaults["qc"]):
        """ Fetches and preprocesses roi data

        Parameters
        ----------
        datasetdir: string
            path to the folder in which to save the data
        metrics: list of strings, see default
            metrics to fetch
        roi_types: list of strings, default ["cortical"]
            type of rois to fetch. Must be one of "cortical", "subcortical"
            and "other"
        test_size: float, default 0.2
            proportion of the dataset to keep for testing. Preprocessing models
            will only be fitted on the training part and applied to the test
            set. You can specify not to use a testing set by setting it to 0
        seed: int, default 42
            random seed to split the data into train / test
        return_data: bool, default False
            If false, saves the data in the specified folder, and return the
            path. Otherwise, returns the preprocessed data and the
            corresponding subjects
        z_score: bool, default True
            wether or not to transform the data into z_scores, meaning
            standardizing and scaling it
        adjust_sites: bool, default True
            wether or not the correct site effects via the Combat algorithm
        residualize_by: dict, see default
            variables to residualize the data. Two keys, "continuous" and
            "discrete", and the values are a list of the variable names
        qc: dict, see default
            keys are the name of the features the control on, values are the
            requirements on their values (see the function apply_qc)

        Returns
        -------
        item: namedtuple
            a named tuple containing 'train_input_path', 'train_metadata_path',
            and 'test_input_path', 'test_metadata_path' if test_size > 0
        X_train: numpy array,
            Training data, if return_data is True
        X_test: numpy array,
            Test data, if return_data is True and test_size > 0
        subj_train: numpy array,
            Training subjects, if return_data is True
        subj_test: numpy array,
            Test subjects, if return_data is True and test_size > 0
        """
        clinical_prefix = "bloc-clinical_score-"
        roi_prefix = "bloc-t1w_roi-"
        subject_column_name = "participant_id"
        path = os.path.join(datasetdir, "rois_X_train.npy")
        meta_path = os.path.join(datasetdir, "rois_X_train.tsv")
        path_test = None
        meta_path_test = None
        if test_size > 0:
            path_test = os.path.join(datasetdir, "rois_X_test.npy")
            meta_path_test = os.path.join(datasetdir, "rois_X_test.tsv")

        if not os.path.isfile(path):
            data = pd.read_csv(files["stratification"], sep="\t")
            roi_mapper = pd.read_csv(files["rois_mapper"], sep="\t")

            # ROI selection
            roi_label_range = pd.Series([False] * len(roi_mapper))
            for roi_type in roi_types:
                if roi_type == "cortical":
                    roi_label_range = roi_label_range | (
                        (roi_mapper["labels"] > 11000) &
                        (roi_mapper["labels"] < 13000))
                elif roi_type == "subcortical":
                    roi_label_range = roi_label_range | (
                        roi_mapper["labels"] > 13000)
                elif roi_type == "other":
                    roi_label_range = roi_label_range | (
                        roi_mapper["labels"] < 11000)
                else:
                    raise ValueError("Roi types must be either 'cortical', \
                        'subcortical' or 'other'")
            roi_labels = roi_mapper.loc[roi_label_range, "labels"]

            # Feature selection
            features_list = []
            for column in data.columns:
                if column.startswith(roi_prefix):
                    roi = int(column.split(":")[1].split("_")[0])
                    metric = column.split("-")[-1]
                    if roi in roi_labels.values and metric in metrics:
                        features_list.append(column.replace(roi_prefix, ""))
            data_train = apply_qc(data, clinical_prefix, qc).sort_values(
                subject_column_name)
            data_train.columns = [elem.replace(roi_prefix, "")
                                  for elem in data_train.columns]
            X_train = data_train[features_list].copy()

            # Splits in train and test and removes nans
            if test_size > 0:
                X_train, X_test, data_train, data_test = train_test_split(
                    X_train, data_train, test_size=test_size,
                    random_state=seed)
                na_idx_test = (X_test.isna().sum(1) == 0)
                X_test = X_test[na_idx_test]
                data_test = data_test[na_idx_test]
                subj_test = data_test[subject_column_name].values
            na_idx_train = (X_train.isna().sum(1) == 0)
            X_train = X_train[na_idx_train]
            data_train = data_train[na_idx_train]
            subj_train = data_train[subject_column_name].values
            cols = X_train.columns

            # Correction for site effects
            if adjust_sites:
                for metric in metrics:
                    adjuster = fortin_combat()
                    features = [feature for feature in features_list
                                if metric in feature]
                    X_train[features] = adjuster.fit_transform(
                        X_train[features],
                        data_train[["{}{}".format(
                            clinical_prefix, site_column_name)]],
                        data_train[["{}{}".format(clinical_prefix, f)
                                    for f in residualize_by["discrete"]]],
                        data_train[["{}{}".format(clinical_prefix, f)
                                    for f in residualize_by["continuous"]]])

                    _path = os.path.join(
                        datasetdir, "rois_combat_{0}.pkl".format(metric))
                    with open(_path, "wb") as of:
                        pickle.dump(adjuster, of)

                    if test_size > 0:
                        X_test[features] = adjuster.transform(
                            X_test[features],
                            data_test[["{}{}".format(
                                clinical_prefix, site_column_name)]],
                            data_test[["{}{}".format(clinical_prefix, f)
                                       for f in residualize_by["discrete"]]],
                            data_test[["{}{}".format(clinical_prefix, f)
                                       for f in residualize_by["continuous"]]])

            # Standardizes
            if z_score:
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                _path = os.path.join(datasetdir, "rois_scaler.pkl")
                with open(_path, "wb") as f:
                    pickle.dump(scaler, f)
                if test_size > 0:
                    X_test = scaler.transform(X_test)
            else:
                X_train = X_train.values
                if test_size > 0:
                    X_test = X_test.values

            # Residualizes and scales
            if residualize_by is not None or len(residualize_by) > 0:
                regressor = LinearRegression()
                y_train = np.concatenate([
                    data_train[["{}{}".format(clinical_prefix, f)
                                for f in residualize_by["continuous"]]].values,
                    OneHotEncoder(sparse=False).fit_transform(
                        data_train[["{}{}".format(clinical_prefix, f)
                                    for f in residualize_by["discrete"]]])
                ], axis=1)
                regressor.fit(y_train, X_train)
                X_train = X_train - regressor.predict(y_train)
                _path = os.path.join(datasetdir, "rois_residualizer.pkl")
                with open(_path, "wb") as f:
                    pickle.dump(regressor, f)

                if test_size > 0:
                    y_test = np.concatenate([
                        data_test[[
                            "{}{}".format(clinical_prefix, f)
                            for f in residualize_by["continuous"]]].values,
                        OneHotEncoder(sparse=False).fit_transform(
                            data_test[["{}{}".format(clinical_prefix, f)
                                       for f in residualize_by["discrete"]]])
                    ], axis=1)
                    X_test = X_test - regressor.predict(y_test)

            # Return data and subjects
            X_train_df = pd.DataFrame(data=X_train, columns=cols)
            X_train_df.insert(0, subject_column_name, subj_train)
            X_test_df = None
            if test_size > 0:
                X_test_df = pd.DataFrame(data=X_test, columns=cols)
                X_test_df.insert(0, subject_column_name, subj_test)

            # Saving
            np.save(path, X_train)
            X_train_df.to_csv(meta_path, index=False, sep="\t")
            if test_size > 0:
                np.save(path_test, X_test)
                X_test_df.to_csv(meta_path_test, index=False, sep="\t")

        if return_data:
            X_train = np.load(path)
            subj_train = pd.read_csv(meta_path, sep="\t")[
                subject_column_name].values
            X_test, subj_test = (None, None)
            if test_size > 0:
                X_test = np.load(path_test)
                subj_test = pd.read_csv(meta_path_test, sep="\t")[
                    subject_column_name].values
            return X_train, X_test, subj_train, subj_test
        else:
            return Item(train_input_path=path, test_input_path=path_test,
                        train_metadata_path=meta_path,
                        test_metadata_path=meta_path_test)

    return fetch_rois


def fetch_surface_wrapper(hemisphere, datasetdir=SAVING_FOLDER,
                          files=FILES, cohort=COHORT_NAME,
                          site_column_name="t1:site",
                          defaults=DEFAULTS["surface"]):
    """ Fetcher wrapper for surface data

    Parameters
    ----------
    hemisphere: string
        name of the hemisphere data fetcher, one of "rh" or "lh"
    datasetdir: string, default SAVING_FOLDER
        path to the folder in which to save the data
    files: dict, default FILES
        contains the paths to the different files
    cohort: string, default COHORT_NAME,
        name of the cohort
    site_columns_name: string, default "t1:site"
        name of the column containing the site of MRI acquisition
    defaults: dict, default DEFAULTS
        default values for the wrapped function

    Returns
    -------
    fetcher: function
        corresponding fetcher

    """
    assert(hemisphere in ["rh", "lh"])
    fetcher_name = "fetcher_surface_{}_{}".format(hemisphere, cohort)

    # @Fetchers.register
    def fetch_surface(
            metrics=defaults["metrics"],
            test_size=defaults["test_size"], seed=defaults["seed"],
            return_data=defaults["return_data"],
            z_score=defaults["z_score"], adjust_sites=defaults["adjust_sites"],
            residualize_by=defaults["residualize_by"], qc=defaults["qc"]):
        """ Fetches and preprocesses surface data

        Parameters
        ----------
        metrics: list of strings, see defaults
            metrics to fetch
        test_size: float, default 0.2
            proportion of the dataset to keep for testing. Preprocessing models
            will only be fitted on the training part and applied to the test
            set. You can specify not to use a testing set by setting it to 0
        seed: int, default 42
            random seed to split the data into train / test
        return_data: bool, default False
            If false, saves the data in the specified folder, and return the
            path. Otherwise, returns the preprocessed data and the
            corresponding subjects
        z_score: bool, default True
            wether or not to transform the data into z_scores, meaning
            standardizing and scaling it
        adjust_sites: bool, default True
            wether or not the correct site effects via the Combat algorithm
        residualize_by: dict, see default
            variables to residualize the data. Two keys, "continuous" and
            "discrete", and the values are a list of the variable names
        qc: dict, see default
            keys are the name of the features the control on, values are the
            requirements on their values (see the function apply_qc)

        Returns
        -------
        item: namedtuple
            a named tuple containing 'train_input_path', 'train_metadata_path',
            and 'test_input_path', 'test_metadata_path' if test_size > 0
        X_train: numpy array,
            Training data, if return_data is True
        X_test: numpy array,
            Test data, if return_data is True and test_size > 0
        subj_train: numpy array,
            Training subjects, if return_data is True
        subj_test: numpy array,
            Test subjects, if return_data is True and test_size > 0
        """

        clinical_prefix = "bloc-clinical_score-"

        surf_prefix = "bloc-t1w_hemi-{}_metric".format(hemisphere)

        data = pd.read_csv(files["clinical_surface"], sep="\t").drop(
            columns=["bloc-t1w_hemi-lh_metric-area",
                     "bloc-t1w_hemi-rh_metric-area"])

        # Feature selection
        features_list = []
        for metric in metrics:
            for column in data.columns:
                if column.startswith(surf_prefix):
                    m = column.split('-')[-1]
                    if m == metric:
                        features_list.append(column)

        data_train = apply_qc(data, clinical_prefix, qc).sort_values(
            "participant_id")

        # Loads surface data
        n_vertices = len(
            surface_loader(data_train[features_list[0]].iloc[0]).get_data())
        X_train = np.zeros((len(data_train), n_vertices, len(features_list)))
        for i in range(len(data_train)):
            for j, feature in enumerate(features_list):
                path = data_train[feature].iloc[i]
                if not pd.isnull([path]):
                    X_train[i, :, j] = surface_loader(
                        path).get_data().squeeze()

        # Splits in train and test and removes nans
        if test_size > 0:
            X_train, X_test, data_train, data_test = train_test_split(
                X_train, data_train, test_size=test_size, random_state=seed)

            na_idx_test = (np.isnan(X_test).sum((1, 2)) == 0)
            X_test = X_test[na_idx_test]
            data_test = data_test[na_idx_test]
            if return_data:
                subj_test = data_test["participant_id"].values

        na_idx_train = (np.isnan(X_train).sum((1, 2)) == 0)

        X_train = X_train[na_idx_train]
        data_train = data_train[na_idx_train]
        if return_data:
            subj_train = data_train["participant_id"].values

        # Applies feature-wise preprocessing
        for i, feature in enumerate(features_list):
            # Correction for site effects
            if adjust_sites:
                non_zeros_idx = (X_train[:, :, i] > 0).sum(0) >= 1
                adjuster = fortin_combat()
                X_train[:, non_zeros_idx, i] = adjuster.fit_transform(
                    X_train[:, non_zeros_idx, i],
                    data_train[["{}{}".format(
                        clinical_prefix, site_column_name)]],
                    data_train[["{}{}".format(clinical_prefix, f)
                                for f in residualize_by["discrete"]]],
                    data_train[["{}{}".format(clinical_prefix, f)
                                for f in residualize_by["continuous"]]])

                path = os.path.join(
                    datasetdir,
                    "surface_{}_combat_feature{}.pkl".format(hemisphere, i))
                with open(path, "wb") as f:
                    pickle.dump(adjuster, f)

                if test_size > 0:
                    X_test[:, non_zeros_idx, i] = adjuster.transform(
                        X_test[:, non_zeros_idx, i],
                        data_test[["{}{}".format(
                            clinical_prefix, site_column_name)]],
                        data_test[["{}{}".format(clinical_prefix, f)
                                   for f in residualize_by["discrete"]]],
                        data_test[["{}{}".format(clinical_prefix, f)
                                   for f in residualize_by["continuous"]]])

            # Standardizes and scales
            if z_score:
                scaler = RobustScaler()
                X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])

                path = os.path.join(
                    datasetdir,
                    "surface_{}_scaler_feature{}.pkl".format(hemisphere, i))
                with open(path, "wb") as f:
                    pickle.dump(scaler, f)
                if test_size > 0:
                    X_test[:, :, i] = scaler.transform(X_test[:, :, i])

            # Residualizes
            if residualize_by is not None or len(residualize_by) > 0:
                regressor = LinearRegression()
                y_train = np.concatenate([
                    data_train[["{}{}".format(clinical_prefix, f)
                                for f in residualize_by["continuous"]]].values,
                    OneHotEncoder(sparse=False).fit_transform(
                        data_train[["{}{}".format(clinical_prefix, f)
                                    for f in residualize_by["discrete"]]])
                ], axis=1)
                regressor.fit(y_train, X_train[:, :, i])
                X_train[:, :, i] = X_train[:, :, i] - regressor.predict(
                    y_train)
                path = os.path.join(
                    datasetdir,
                    "surface_{}_residualizer_feature{}.pkl".format(
                        hemisphere, i))
                with open(path, "wb") as f:
                    pickle.dump(regressor, f)

                if test_size > 0:
                    y_test = np.concatenate([
                        data_test[["{}{}".format(clinical_prefix, f)
                                   for f in residualize_by["continuous"]]
                                  ].values,
                        OneHotEncoder(sparse=False).fit_transform(
                            data_test[["{}{}".format(clinical_prefix, f)
                                       for f in residualize_by["discrete"]]])
                    ], axis=1)
                    X_test[:, :, i] = X_test[:, :, i] - regressor.predict(
                        y_test)

        # Returns data and subjects
        if return_data:
            if test_size > 0:
                return X_train, X_test, subj_train, subj_test
            return X_train, subj_train

        # Saving
        path = os.path.join(
            datasetdir, "surface_{}_X_train.npy".format(hemisphere))
        np.save(path, X_train)
        if test_size > 0:
            path_test = os.path.join(
                datasetdir, "surface_{}_X_test.npy".format(hemisphere))
            np.save(path_test, X_test)
            return path, path_test

        return path
    return fetch_surface


def fetch_genetic_wrapper(datasetdir=SAVING_FOLDER, files=FILES,
                          cohort=COHORT_NAME, defaults=DEFAULTS['genetic']):
    """ Fetcher wrapper for genetic data

    Parameters
    ----------
    datasetdir: string, default SAVING_FOLDER
        path to the folder in which to save the data
    files: dict, default FILES
        contains the paths to the different files
    cohort: string, default COHORT_NAME,
        name of the cohort
    defaults: dict, default DEFAULTS
        default values for the wrapped function

    Returns
    -------
    fetcher: function
        corresponding fetcher

    """

    fetcher_name = "fetcher_genetic_{}".format(cohort)

    # @Fetchers.register
    def fetch_genetic(
            scores=defaults["scores"], test_size=defaults["test_size"],
            seed=defaults["seed"], return_data=defaults["return_data"],
            z_score=defaults["z_score"], qc=defaults["qc"]):
        """ Fetches and preprocesses genetic data

        Parameters
        ----------
        scores: list of strings, see defaults
            scores to fetch, None mean it fetches all the available scores
        test_size: float, see defaults
            proportion of the dataset to keep for testing. Preprocessing models
            will only be fitted on the training part and applied to the test
            set. You can specify not to use a testing set by setting it to 0
        seed: int, see default
            random seed to split the data into train / test
        return_data: bool, default False
            If false, saves the data in the specified folder, and return the
            path. Otherwise, returns the preprocessed data and the
            corresponding subjects
        z_score: bool, see defaults
            wether or not to transform the data into z_scores, meaning
            standardizing and scaling it
        qc: dict, see defaults
            keys are the name of the features the control on, values are the
            requirements on their values (see the function apply_qc)

        Returns
        -------
        item: namedtuple
            a named tuple containing 'train_input_path', 'train_metadata_path',
            and 'test_input_path', 'test_metadata_path' if test_size > 0
        X_train: numpy array
            Training data, if return_data is True
        X_test: numpy array
            Test data, if return_data is True and test_size > 0
        subj_train: numpy array
            Training subjects, if return_data is True
        subj_test: numpy array
            Test subjects, if return_data is True and test_size > 0
        """

        clinical_prefix = "bloc-clinical_score-"
        genetic_prefix = "bloc-genetic_score-"
        subject_column_name = "participant_id"
        path = os.path.join(datasetdir, "genetic_X_train.npy")
        meta_path = os.path.join(datasetdir, "genetic_X_train.tsv")
        path_test = None
        meta_path_test = None
        if test_size > 0:
            path_test = os.path.join(datasetdir, "genetic_X_test.npy")
            meta_path_test = os.path.join(datasetdir, "genetic_X_test.tsv")

        if not os.path.isfile(path):

            data = pd.read_csv(files["stratification"], sep="\t")

            # Feature selection
            features_list = []
            for column in data.columns:
                if column.startswith(genetic_prefix):
                    score = column.split("-")[-1]
                    if scores is not None and score in scores:
                        features_list.append(
                            column.replace(genetic_prefix, ""))
                    elif scores is None:
                        features_list.append(
                            column.replace(genetic_prefix, ""))
            data_train = apply_qc(data, clinical_prefix, qc).sort_values(
                subject_column_name)
            data_train.columns = [elem.replace(genetic_prefix, "")
                                  for elem in data_train.columns]
            X_train = data_train[features_list].copy()

            # Splits in train and test and removes nans
            if test_size > 0:
                X_train, X_test, data_train, data_test = train_test_split(
                    X_train, data_train, test_size=test_size,
                    random_state=seed)
                na_idx_test = (X_test.isna().sum(1) == 0)
                X_test = X_test[na_idx_test]
                data_test = data_test[na_idx_test]
                subj_test = data_test[subject_column_name].values
            na_idx_train = (X_train.isna().sum(1) == 0)
            X_train = X_train[na_idx_train]
            data_train = data_train[na_idx_train]
            subj_train = data_train[subject_column_name].values
            cols = X_train.columns

            # Standardizes and scales
            if z_score:
                scaler = RobustScaler()
                X_train = scaler.fit_transform(X_train)
                _path = os.path.join(datasetdir, "genetic_scaler.pkl")
                with open(_path, "wb") as f:
                    pickle.dump(scaler, f)
                if test_size > 0:
                    X_test = scaler.transform(X_test)
            else:
                X_train = X_train.values
                if test_size > 0:
                    X_test = X_test.values

            # Return data and subjects
            X_train_df = pd.DataFrame(data=X_train, columns=cols)
            X_train_df.insert(0, subject_column_name, subj_train)
            X_test_df = None
            if test_size > 0:
                X_test_df = pd.DataFrame(data=X_test, columns=cols)
                X_test_df.insert(0, subject_column_name, subj_test)

            # Saving
            np.save(path, X_train)
            X_train_df.to_csv(meta_path, index=False, sep="\t")
            if test_size > 0:
                np.save(path_test, X_test)
                X_test_df.to_csv(meta_path_test, index=False, sep="\t")

        if return_data:
            X_train = np.load(path)
            subj_train = pd.read_csv(meta_path, sep="\t")[
                subject_column_name].values
            X_test, subj_test = (None, None)
            if test_size > 0:
                X_test = np.load(path_test)
                subj_test = pd.read_csv(meta_path_test, sep="\t")[
                    subject_column_name].values
            return X_train, X_test, subj_train, subj_test
        else:
            return Item(train_input_path=path, test_input_path=path_test,
                        train_metadata_path=meta_path,
                        test_metadata_path=meta_path_test)
    return fetch_genetic


def make_fetchers(datasetdir=SAVING_FOLDER):

    return {
        "clinical": fetch_clinical_wrapper(datasetdir=datasetdir),
        "rois": fetch_rois_wrapper(datasetdir=datasetdir),
        "surface-rh": fetch_surface_wrapper(hemisphere="rh",
                                            datasetdir=datasetdir),
        "surface-lh": fetch_surface_wrapper(hemisphere="lh",
                                            datasetdir=datasetdir),
        "genetic": fetch_genetic_wrapper(datasetdir=datasetdir),
    }


def fetch_multiblock_wrapper(datasetdir=SAVING_FOLDER, files=FILES,
                             cohort=COHORT_NAME,
                             subject_column_name="subjects",
                             defaults=DEFAULTS["multiblock"],
                             make_fetchers_func=make_fetchers):
    """ Fetcher wrapper for multiblock data

    Parameters
    ----------
    datasetdir: string, default SAVING_FOLDER
        path to the folder in which to save the data
    files: dict, default FILES
        contains the paths to the different files
    cohort: string, default COHORT_NAME,
        name of the cohort
    subject_columns_name: string, default "subjects"
        name of the column containing the subjects id
    defaults: dict, default DEFAULTS
        default values for the wrapped function
    make_fetchers_func: function, default make_fetchers
        function to build the fetchers from their wrappers.
        Must return a dict containing as keys the name of the
        channels, and values the corresponding fetcher

    Returns
    -------
    fetcher: function
        corresponding fetcher

    """

    fetcher_name = "fetcher_multiblock_{}".format(cohort)
    FETCHERS = make_fetchers_func(datasetdir)

    # @Fetchers.register
    def fetch_multiblock(
            blocks=defaults["blocks"],
            test_size=defaults["test_size"], seed=defaults["seed"],
            qc=defaults["qc"],
            **kwargs):
        """ Fetches and preprocesses multi block data

        Parameters
        ----------
        blocks: list of strings, see default
            blocks of data to fetch, all must be in the key list of FETCHERS
        test_size: float, default 0.2
            proportion of the dataset to keep for testing. Preprocessing models
            will only be fitted on the training part and applied to the test
            set. You can specify not to use a testing set by setting it to 0
        seed: int, default 42
            random seed to split the data into train / test
        qc: dict, see default
            keys are the name of the features the control on, values are the
            requirements on their values (see the function apply_qc)
        kwargs: dict
            additional arguments to be passed to each fetcher indivudally.
            Keys are the name of the fetchers, and values are a dictionnary
            containing arguments and the values for this fetcher

        Returns
        -------
        item: namedtuple
            a named tuple containing 'train_input_path', 'train_metadata_path',
            and 'test_input_path', 'test_metadata_path' if test_size > 0
        """

        path = os.path.join(datasetdir, "multiblock_X_train.npz")
        metadata_path = os.path.join(datasetdir, "metadata_train.tsv")
        path_test = None
        metadata_path_test = None
        if test_size > 0:
            path_test = os.path.join(datasetdir, "multiblock_X_test.npz")
            metadata_path_test = os.path.join(
                datasetdir, "metadata_test.tsv")

        if not os.path.isfile(path):
            X_train = {}
            subj_train = {}
            if test_size > 0:
                X_test = {}
                subj_test = {}
            for block in blocks:
                assert block in FETCHERS.keys()
                if block in kwargs.keys():
                    local_kwargs = kwargs[block]
                    # Impose to have the same qc steps and splitting train/test
                    # over all the blocks to have the same subjects
                    for key, value in local_kwargs.items():
                        if key in ["qc", "test_size", "seed"]:
                            del local_kwargs[key]
                else:
                    local_kwargs = {}
                new_X_train, new_X_test, new_subj_train, new_subj_test = \
                    FETCHERS[block](
                        qc=qc, test_size=test_size, seed=seed,
                        return_data=True, **local_kwargs)
                if test_size > 0:
                    X_test[block] = new_X_test
                    subj_test[block] = new_subj_test
                X_train[block] = new_X_train
                subj_train[block] = new_subj_train

            # Remove subjects that arent in all the channels
            common_subjects_train = list(
                set.intersection(*map(set, subj_train.values())))
            for block in blocks:
                subjects = subj_train[block]
                assert(len(subjects) == len(X_train[block]))
                idx_to_keep = [
                    _idx for _idx in range(len(subjects))
                    if subjects[_idx] in common_subjects_train]
                X_train[block] = X_train[block][idx_to_keep]

            if test_size > 0:
                common_subjects_test = list(
                    set.intersection(*map(set, subj_test.values())))
                for block in blocks:
                    subjects = subj_test[block]
                    assert(len(subjects) == len(X_test[block]))
                    idx_to_keep = [
                        _idx for _idx in range(len(subjects))
                        if subjects[_idx] in common_subjects_test]
                    X_test[block] = X_test[block][idx_to_keep]

            # Loads metadata
            clinical_prefix = "bloc-clinical_score-"
            metadata_cols = ["participant_id", "labels", "subgroups"]
            metadata = pd.read_csv(files["stratification"], sep="\t")
            clinical_cols = ["participant_id"]
            clinical_cols += [col for col in metadata.columns
                              if col.startswith(clinical_prefix)]
            metadata = metadata[clinical_cols]
            metadata.columns = [elem.replace(clinical_prefix, "")
                                for elem in metadata.columns]
            metadata = metadata[metadata_cols]
            metadata_train = metadata[
                metadata[subject_column_name].isin(common_subjects_train)]
            if test_size > 0:
                metadata_test = metadata[
                    metadata[subject_column_name].isin(common_subjects_test)]

            # Saving
            np.savez(path, **X_train)
            metadata_train.to_csv(metadata_path, index=False, sep="\t")
            if test_size > 0:
                np.savez(path_test, **X_test)
                metadata_test.to_csv(metadata_path_test, index=False, sep="\t")

        return Item(train_input_path=path, test_input_path=path_test,
                    train_metadata_path=metadata_path,
                    test_metadata_path=metadata_path_test)

    return fetch_multiblock


WRAPPERS = {
    "clinical": fetch_clinical_wrapper,
    "rois": fetch_rois_wrapper,
    "genetic": fetch_genetic_wrapper,
    "surface": fetch_surface_wrapper,
    "multiblock": fetch_multiblock_wrapper,
}


def fetch_multiblock_euaims(datasetdir=SAVING_FOLDER, fetchers=make_fetchers,
                            surface=False):
    if surface:
        DEFAULTS["multiblock"]["blocks"] = ["clinical", "surface-lh",
                                            "surface-rh", "genetic"]
    else:
        DEFAULTS["multiblock"]["blocks"] = ["clinical", "rois", "genetic"]
    return WRAPPERS["multiblock"](
        datasetdir=datasetdir, files=FILES, cohort=COHORT_NAME,
        subject_column_name="participant_id", defaults=DEFAULTS["multiblock"],
        make_fetchers_func=make_fetchers)()


def inverse_normalization(data, scalers):
    """ De-normalize a dataset.
    """
    for scaler_path in scalers:
        with open(scaler_path, "rb") as of:
            scaler = pickle.load(of)
        data = scaler.inverse_transform(data)
    return data

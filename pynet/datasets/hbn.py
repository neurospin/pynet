# -*- coding: utf-8 -*-
########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
########################################################################

"""
Module provides functions to prepare different datasets from HBN.
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from pynet.datasets import Fetchers
from neurocombat_sklearn import CombatModel as fortin_combat
from nilearn.surface import load_surf_data


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

FOLDER = "/neurospin/brainomics/2020_deepint/data/"

SAVING_FOLDER = "/neurospin/brainomics/2020_deepint/preprocessed_data/"

FILES = {
    "clinical": os.path.join(FOLDER, "HBN_clinical.tsv"),
    "rois_mapper": os.path.join(FOLDER, "HBN_rois.tsv"),
    "clinical_rois": os.path.join(FOLDER, "HBN_stratification.tsv"),
    "clinical_subgroups_full": os.path.join(FOLDER, "HBN_subgroups.tsv"),
    "clinical_surface": os.path.join(FOLDER, "HBN_surf_stratification.tsv"),
    "clinical_subgroups": os.path.join(FOLDER, "HBN_subgroups_angeline.csv"),
}

logger = logging.getLogger("pynet")


def fetch_clinical(datasetdir=SAVING_FOLDER, z_score=True,
    drop_cols=["EID", "Study Site", "Age", "Sex", "WISC_FSIQ", "MRI", "EULER"]):

    data = pd.read_table(FILES["clinical"])

    data.drop(columns=drop_cols, inplace=True)

    if z_score:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        with open("clinical_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    else:
        data = data.values

    path = os.path.join(datasetdir, "clinical.npy")
    np.save(path, data)

    return path

def apply_qc(data, prefix, qc, na_method='remove'):
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
                raise ValueError("The relationship {} provided is not a valid one".format(relation))
            else:
                new_idx = relation_mapper[relation](data["{}-{}".format(prefix, name)], value)
                idx_to_keep = idx_to_keep & new_idx
    if na_method == 'remove':
        idx_to_keep = idx_to_keep & (data.isna().sum(1) == 0)
    return data[idx_to_keep]

def fetch_rois(datasetdir=SAVING_FOLDER,
    metrics=["lgi:avg", "thick:avg", "surf:area"],
    roi_types=["cortical"], z_score=True, adjust_sites=True,
    residualize_by={"continuous": ["age", "wisc:fsiq"], "discrete":["sex"]},
    qc={"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217}, "mri": {"eq": 1}},
    na_method="remove", test_size=0.2, seed=42):

    clinical_prefix = "bloc-clinical_score"

    roi_prefix = "bloc-t1w_roi"
    
    data = pd.read_table(FILES["clinical_rois"])
    roi_mapper = pd.read_table(FILES["rois_mapper"])

    # ROI selection
    roi_label_range = pd.Series([False] * len(roi_mapper))
    for roi_type in roi_types:
        if roi_type == "cortical":
            roi_label_range = roi_label_range | ( 
                (roi_mapper["labels"] > 11000) & (roi_mapper["labels"] < 13000) )
        elif roi_type == "subcortical":
            roi_label_range = roi_label_range | ( roi_mapper["labels"] > 13000 )
        elif roi_type == "other":
            roi_label_range = roi_label_range | ( roi_mapper["labels"] < 11000 )
        else:
            raise ValueError("Roi types must be either 'cortical', 'subcortical' or 'other'")
    

    roi_labels = roi_mapper.loc[roi_label_range, "labels"]
    print(roi_labels)

    # Feature selection
    features_list = []
    for column in data.columns:
        if column.startswith(roi_prefix):
            roi = int(column.split(":")[1].split("_")[0])
            metric = column.split("-")[-1]
            if roi in roi_labels.values and metric in metrics:
                features_list.append(column)
    data_train = apply_qc(data, clinical_prefix, qc, na_method).sort_values("participant_id")
    print(data_train.shape)

    X_train = data_train[features_list].copy()

    if test_size != 0:
        X_train, X_test, data_train, data_test = train_test_split(
            X_train, data_train, test_size=test_size, random_state=seed)

    # Correction for site effects
    if adjust_sites:
        for metric in metrics:
            adjuster = fortin_combat()
            features = [feature for feature in features_list if metric in feature]
            X_train[features] = adjuster.fit_transform(X_train[features],
                data_train[["{}-study site".format(clinical_prefix)]],
                data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])
            
            path = os.path.join(datasetdir, "rois_combat.pkl")
            with open(path, "wb") as f:
                pickle.dump(adjuster, f)

            if test_size != 0:
                X_test[features] = adjuster.transform(X_test,
                data_test[["{}-study site".format(clinical_prefix)]],
                data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])

    
    # Standerdize
    if z_score:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        path = os.path.join(datasetdir, "rois_scaler.pkl")
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
        if test_size != 0:
            X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        if test_size != 0:
            X_test = X_test.values

    # Residualize
    if residualize_by is not None or len(residualize_by) != 0:
        regressor = LinearRegression()
        y_train = np.concatenate([
            data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
            OneHotEncoder(sparse=False).fit_transform(
                data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
        ], axis=1)
        print(y_train.shape)
        t = time.time()
        regressor.fit(y_train, X_train)
        X_train = X_train - regressor.predict(y_train)
        print(time.time() - t)
        path = os.path.join(datasetdir, "rois_residualizer.pkl")
        with open(path, "wb") as f:
            pickle.dump(regressor, f)

        if test_size != 0:
            y_test = np.concatenate([
            data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
                OneHotEncoder(sparse=False).fit_transform(
                data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
            ], axis=1)
            X_test = X_test - regressor.predict(y_test)
        
    # Saving
    path = os.path.join(datasetdir, "HBN_rois_X_train.npy")
    np.save(path, data)

    if test_size != 0:
        path_test = os.path.join(datasetdir, "HBN_rois_X_test.npy")
        np.save(path_test, data)
        return path, path_test

    return path

def fetch_surface(hemisphere, datasetdir=SAVING_FOLDER,
    metrics=["pial_lgi", "thickness"],
    roi_types=["cortical"], z_score=True, adjust_sites=True,
    residualize_by={"continuous": ["age", "wisc:fsiq"], "discrete":["sex"]},
    qc={"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217}, "mri": {"eq": 1}},
    na_method="remove", test_size=0.2, seed=42):
    
    assert(hemisphere in ["rh", "lh"])

    clinical_prefix = "bloc-clinical_score"

    surf_prefix = "bloc-t1w_hemi-{}_metric".format(hemisphere)
    
    data = pd.read_table(FILES["clinical_surface"]).drop(columns=["bloc-t1w_hemi-lh_metric-area", "bloc-t1w_hemi-rh_metric-area"])

    features_list = []
    for column in data.columns:
        if column.startswith(surf_prefix):
            metric = column.split('-')[-1]
            if metric in metrics:
                features_list.append(column)
    print(features_list)
    data_train = apply_qc(data, clinical_prefix, qc, na_method).sort_values("participant_id")
    print(data_train.shape)

    n_vertices = len(load_surf_data(data_train[features_list[0]].iloc[0]))
    X_train = np.zeros((len(data_train), n_vertices, len(features_list)))
    for i in range(len(data_train)):
        for j, feature in enumerate(features_list):
            path = data_train[feature].iloc[i]
            X_train[i,:, j] = load_surf_data(path)   

    if test_size != 0:
        X_train, X_test, data_train, data_test = train_test_split(
            X_train, data_train, test_size=test_size, random_state=seed)

    
    for i, feature in enumerate(features_list):
        print(feature)
        # Correction for site effects
        if adjust_sites:
            # print(((X_train[:, :, i] == 0).sum(0) == len(X_train)).sum())
            # print(((X_test[:, :, i] == 0).sum(0) == len(X_test)).sum())
            non_zeros_idx = (X_train[:, :, i] != 0).sum(0) >= 1# | ((X_test[:, :, i] != 0).sum(0) >= 1)
            print(X_train.shape[1] - non_zeros_idx.sum())
            adjuster = fortin_combat()
            X_train[:, non_zeros_idx, i] = adjuster.fit_transform(X_train[:, non_zeros_idx, i],
                data_train[["{}-study site".format(clinical_prefix)]],
                data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])
            
            path = os.path.join(datasetdir, "surface_combat_feature{}.pkl".format(i))
            with open(path, "wb") as f:
                pickle.dump(adjuster, f)

            if test_size != 0:
                X_test[:, non_zeros_idx, i] = adjuster.transform(X_test[:, non_zeros_idx, i],
                data_test[["{}-study site".format(clinical_prefix)]],
                data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])
        
        # Standardize
        if z_score:
            scaler = StandardScaler()
            X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])

            path = os.path.join(datasetdir, "surface_scaler_feature{}.pkl".format(i))            
            with open(path, "wb") as f:
                pickle.dump(scaler, f)
            if test_size != 0:
                X_test[:, :, i] = scaler.transform(X_test[:, :, i])

        # Residualize
        if residualize_by is not None or len(residualize_by) != 0:
            regressor = LinearRegression()
            y_train = np.concatenate([
                data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
                OneHotEncoder(sparse=False).fit_transform(
                    data_train[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
            ], axis=1)
            print(y_train.shape)
            t = time.time()
            regressor.fit(y_train, X_train[:, :, i])
            X_train[:, :, i] = X_train[:, :, i] - regressor.predict(y_train)
            print(time.time() - t)
            path = os.path.join(datasetdir, "surface_residualizer_feature{}.pkl".format(i))
            with open(path, "wb") as f:
                pickle.dump(regressor, f)

            if test_size != 0:
                y_test = np.concatenate([
                data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
                    OneHotEncoder(sparse=False).fit_transform(
                    data_test[["{}-{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
                ], axis=1)
                X_test[:, :, i] = X_test[:, :, i] - regressor.predict(y_test)
    
    # Saving
    path = os.path.join(datasetdir, "HBN_surface_X_train.npy")
    np.save(path, data)

    if test_size != 0:
        path_test = os.path.join(datasetdir, "HBN_surface_X_test.npy")
        np.save(path_test, data)
        return path, path_test

    return path

print(fetch_surface('rh'))




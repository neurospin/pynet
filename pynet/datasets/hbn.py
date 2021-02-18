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

SAVING_FOLDER = "/neurospin/brainomics/2020_deepint/preprocessed_data/HBN"

FILES = {
    "clinical": os.path.join(FOLDER, "HBN_clinical.tsv"),
    "rois_mapper": os.path.join(FOLDER, "HBN_rois.tsv"),
    "clinical_rois": os.path.join(FOLDER, "HBN_stratification.tsv"),
    "clinical_subgroups_full": os.path.join(FOLDER, "HBN_subgroups.tsv"),
    "clinical_surface": os.path.join(FOLDER, "HBN_surf_stratification.tsv"),
    "clinical_subgroups": os.path.join(FOLDER, "HBN_subgroups_angeline.csv"),
}

logger = logging.getLogger("pynet")

def apply_qc(data, prefix, qc):
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
                new_idx = relation_mapper[relation](data["{}{}".format(prefix, name)], value)
                idx_to_keep = idx_to_keep & new_idx
    # if na_method == 'remove':
    #     idx_to_keep = idx_to_keep & (data.isna().sum(1) == 0)
    return data[idx_to_keep]

def fetch_clinical(datasetdir=SAVING_FOLDER, z_score=True,
    drop_cols=["study site", "age", "sex", "wisc:fsiq", "mri", "euler"],
    qc={"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217}, "mri": {"eq": 1}},
    test_size=0.2, seed=42, return_data=False):

    data = pd.read_table(FILES["clinical"])

    data_train = apply_qc(data, '', qc).sort_values("EID")

    X_train = data_train.drop(columns=drop_cols)

    if test_size != 0:
        X_train, X_test = train_test_split(X_train,
            test_size=test_size, random_state=seed)

        
        na_idx_test = (X_test.isna().sum(1) == 0)
        X_test = X_test[na_idx_test]
        if return_data:
            subj_test = X_test["EID"].values
        X_test = X_test.drop(columns=["EID"]).values

    na_idx_train = (X_train.isna().sum(1) == 0)
    X_train = X_train[na_idx_train]
    if return_data:
        subj_train = X_train["EID"].values
    X_train = X_train.drop(columns=["EID"]).values
   
    if z_score:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        path = os.path.join(datasetdir, "clinical_scaler.pkl")
        with open(path, "wb") as f:
            pickle.dump(scaler, f)
        if test_size != 0:
            X_test = scaler.transform(X_test)

    if return_data:
        if test_size != 0:
            return X_train, X_test, subj_train, subj_test
        return X_train, subj_train
    path = os.path.join(datasetdir, "HBN_clinical_X_train.npy")
    np.save(path, data)
    if test_size != 0:
        path_test = os.path.join(datasetdir, "HBN_clinical_X_test.npy")
        np.save(path_test, data)
        return path, path_test

    return path

def fetch_rois(datasetdir=SAVING_FOLDER,
    metrics=["lgi:avg", "thick:avg", "surf:area"],
    roi_types=["cortical"], z_score=True, adjust_sites=True,
    residualize_by={"continuous": ["age", "wisc:fsiq"], "discrete":["sex"]},
    qc={"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217}, "mri": {"eq": 1}},
    test_size=0.2, seed=42, return_data=False):

    clinical_prefix = "bloc-clinical_score-"

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
    data_train = apply_qc(data, clinical_prefix, qc).sort_values("participant_id")
    # print(data_train.shape)

    X_train = data_train[features_list].copy()

    if test_size != 0:
        X_train, X_test, data_train, data_test = train_test_split(
            X_train, data_train, test_size=test_size, random_state=seed)

        na_idx_test = (X_test.isna().sum(1) == 0)
        X_test = X_test[na_idx_test]
        data_test = data_test[na_idx_test]
        if return_data:
            subj_test = data_test["participant_id"].values

    na_idx_train = (X_train.isna().sum(1) == 0)
    X_train = X_train[na_idx_train]
    data_train = data_train[na_idx_train]
    if return_data:
        subj_train = data_train["participant_id"].values

    # Correction for site effects
    if adjust_sites:
        for metric in metrics:
            adjuster = fortin_combat()
            features = [feature for feature in features_list if metric in feature]
            X_train[features] = adjuster.fit_transform(X_train[features],
                data_train[["{}study site".format(clinical_prefix)]],
                data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])
            
            path = os.path.join(datasetdir, "rois_combat.pkl")
            with open(path, "wb") as f:
                pickle.dump(adjuster, f)

            if test_size != 0:
                X_test[features] = adjuster.transform(X_test[features],
                data_test[["{}study site".format(clinical_prefix)]],
                data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])

    
    # Standardize
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
            data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
            OneHotEncoder(sparse=False).fit_transform(
                data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
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
            data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
                OneHotEncoder(sparse=False).fit_transform(
                data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
            ], axis=1)
            X_test = X_test - regressor.predict(y_test)
        
    # Saving
    if return_data:
        if test_size != 0:
            return X_train, X_test, subj_train, subj_test
        return X_train, subj_train
    path = os.path.join(datasetdir, "HBN_rois_X_train.npy")
    np.save(path, data)

    if test_size != 0:
        path_test = os.path.join(datasetdir, "HBN_rois_X_test.npy")
        np.save(path_test, data)
        return path, path_test

    return path
# print(fetch_rois())

def fetch_surface(hemisphere):
    assert(hemisphere in ["rh", "lh"])

    def fetch_surface_hemi(datasetdir=SAVING_FOLDER,
        metrics=["pial_lgi", "thickness"], return_data=False,
        roi_types=["cortical"], z_score=True, adjust_sites=True,
        residualize_by={"continuous": ["age", "wisc:fsiq"], "discrete":["sex"]},
        qc={"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217}, "mri": {"eq": 1}},
        test_size=0.2, seed=42):

        clinical_prefix = "bloc-clinical_score-"

        surf_prefix = "bloc-t1w_hemi-{}_metric".format(hemisphere)
        
        data = pd.read_table(FILES["clinical_surface"]).drop(columns=["bloc-t1w_hemi-lh_metric-area", "bloc-t1w_hemi-rh_metric-area"])

        features_list = []
        for metric in metrics:
            for column in data.columns:
                if column.startswith(surf_prefix):
                    m = column.split('-')[-1]
                    if m == metric:
                        features_list.append(column)

        data_train = apply_qc(data, clinical_prefix, qc).sort_values("participant_id")

        n_vertices = len(load_surf_data(data_train[features_list[0]].iloc[0]))
        X_train = np.zeros((len(data_train), n_vertices, len(features_list)))
        for i in range(len(data_train)):
            for j, feature in enumerate(features_list):
                path = data_train[feature].iloc[i]
                if not pd.isnull([path]):
                    X_train[i,:, j] = load_surf_data(path)   

        if test_size != 0:
            X_train, X_test, data_train, data_test = train_test_split(
                X_train, data_train, test_size=test_size, random_state=seed)

            na_idx_test = (np.isnan(X_test).sum((1,2)) == 0)
            X_test = X_test[na_idx_test]
            data_test = data_test[na_idx_test]
            if return_data:
                subj_test = data_test["participant_id"].values

        na_idx_train = (np.isnan(X_train).sum((1, 2)) == 0)
        
        X_train = X_train[na_idx_train]
        data_train = data_train[na_idx_train]
        if return_data:
            subj_train = data_train["participant_id"].values
        
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
                    data_train[["{}study site".format(clinical_prefix)]],
                    data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                    data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])
                
                path = os.path.join(datasetdir, "surface_{}_combat_feature{}.pkl".format(hemisphere, i))
                with open(path, "wb") as f:
                    pickle.dump(adjuster, f)

                if test_size != 0:
                    X_test[:, non_zeros_idx, i] = adjuster.transform(X_test[:, non_zeros_idx, i],
                    data_test[["{}study site".format(clinical_prefix)]],
                    data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]],
                    data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]])
            
            # Standardize
            if z_score:
                scaler = StandardScaler()
                X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i])

                path = os.path.join(datasetdir, "surface_{}_scaler_feature{}.pkl".format(hemisphere, i))            
                with open(path, "wb") as f:
                    pickle.dump(scaler, f)
                if test_size != 0:
                    X_test[:, :, i] = scaler.transform(X_test[:, :, i])

            # Residualize
            if residualize_by is not None or len(residualize_by) != 0:
                regressor = LinearRegression()
                y_train = np.concatenate([
                    data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
                    OneHotEncoder(sparse=False).fit_transform(
                        data_train[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
                ], axis=1)
                print(y_train.shape)
                t = time.time()
                regressor.fit(y_train, X_train[:, :, i])
                X_train[:, :, i] = X_train[:, :, i] - regressor.predict(y_train)
                print(time.time() - t)
                path = os.path.join(datasetdir, "surface_{}_residualizer_feature{}.pkl".format(hemisphere, i))
                with open(path, "wb") as f:
                    pickle.dump(regressor, f)

                if test_size != 0:
                    y_test = np.concatenate([
                    data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["continuous"]]].values,
                        OneHotEncoder(sparse=False).fit_transform(
                        data_test[["{}{}".format(clinical_prefix, f) for f in residualize_by["discrete"]]])
                    ], axis=1)
                    X_test[:, :, i] = X_test[:, :, i] - regressor.predict(y_test)
        
        # Saving
        if return_data:
            if test_size != 0:
                return X_train, X_test, subj_train, subj_test
            return X_train, subj_train
        path = os.path.join(datasetdir, "HBN_surface_{}_X_train.npy".format(hemisphere))
        np.save(path, X_train)

        if test_size != 0:
            path_test = os.path.join(datasetdir, "HBN_surface_{}_X_test.npy".format(hemisphere))
            np.save(path_test, X_test)
            return path, path_test

        return path
    return fetch_surface_hemi

FETCHERS = {
    'clinical': fetch_clinical,
    'rois': fetch_rois,
    'surface-rh': fetch_surface('rh'),
    'surface-lh': fetch_surface('lh'),
}

def fetch_multi_block(blocks=['clinical', 'surface-lh', 'surface-rh'],
    datasetdir=SAVING_FOLDER,
    qc={"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217}, "mri": {"eq": 1}},
    test_size=0.2, seed=42, **kwargs):
    X_train = []
    subj_train = []
    if test_size != 0:
        X_test = []
        subj_test = []
    for block in blocks:
        assert block in ['clinical', 'surface-lh', 'surface-rh', 'rois']
        if block in kwargs.keys():
            local_kwargs = kwargs[block]

            # Impose to have the same qc steps and splitting train/test over all the blocks 
            # to have the same subjects
            for key, value in local_kwargs.items():
                if key in ["datasetdir", "qc", "test_size", "seed"]:
                    del local_kwargs[key]
        else:
            local_kwargs = {}
        if test_size != 0:
            new_X_train, new_X_test, new_subj_train, new_subj_test = FETCHERS[block](datasetdir, qc=qc,
                test_size=test_size, seed=seed, 
                return_data=True, **local_kwargs)
            X_test.append(new_X_test)
            subj_test.append(new_subj_test)
        else:
            new_X_train, new_subj_train  = FETCHERS[block](datasetdir, qc=qc,
                test_size=test_size, seed=seed,
                return_data=True, **local_kwargs)
        X_train.append(new_X_train)
        subj_train.append(new_subj_train)
        print(new_X_train.nbytes)
        print(new_X_train.shape)
    
    # Remove subjects that werent in all the channels
    common_subjects_train = subj_train[0]
    for subjects in subj_train[1:]:
        common_subjects_train = [sub for sub in subjects if sub in common_subjects_train]

    idx_to_keep = []
    for subjects in subj_train:
        new_idx_to_keep = []
        for i, sub in enumerate(subjects):
            if sub in common_subjects_train:
                new_idx_to_keep.append(i)
        idx_to_keep.append(new_idx_to_keep)
    for i in range(len(X_train)):
        X_train[i] = X_train[i][idx_to_keep[i]]
        print(X_train[i].shape)
    
    common_subjects_test = subj_test[0]
    for subjects in subj_test[1:]:
        common_subjects_test = [sub for sub in subjects if sub in common_subjects_test]

    idx_to_keep = []
    for subjects in subj_test:
        new_idx_to_keep = []
        for i, sub in enumerate(subjects):
            if sub in common_subjects_test:
                new_idx_to_keep.append(i)
        idx_to_keep.append(new_idx_to_keep)
    for i in range(len(X_test)):
        X_test[i] = X_test[i][idx_to_keep[i]]
        print(X_test[i].shape)

    path = os.path.join(datasetdir, "HBN_multi-block_X_train.npz")
    np.savez(path, *X_train)
    if test_size != 0:
        path_test = os.path.join(datasetdir, "HBN_multi-block_X_test.npz")
        np.savez(path_test, *X_test)

        return path, path_test
    return path

print(fetch_multi_block())




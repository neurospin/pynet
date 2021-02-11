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
from sklearn.preprocessing import StandardScaler
from pynet.datasets import Fetchers
from neurocombat_sklearn import CombatModel as fortin_combat



# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

FOLDER = "/neurospin/brainomics/2020_deepint/data/"

SAVING_FOLDER = "/neurospin/brainomics/2020_deepint/preprocessed_data/"

FILES = {
    'clinical': os.path.join(FOLDER, 'HBN_clinical.tsv'),
    'rois_mapper': os.path.join(FOLDER, 'HBN_rois.tsv'),
    'clinical_rois': os.path.join(FOLDER, 'HBN_stratification.tsv'),
    'clinical_subgroups_full': os.path.join(FOLDER, 'HBN_subgroups.tsv'),
    'clinical_surface': os.path.join(FOLDER, 'HBN_surf_stratification.tsv'),
    'clinical_subgroups': os.path.join(FOLDER, 'HBN_subgroups_angeline.csv'),
}

logger = logging.getLogger("pynet")


def fetch_clinical(datasetdir=SAVING_FOLDER, z_score=True,
    drop_cols=['EID', 'Study Site', 'Age', 'Sex', 'WISC_FSIQ', 'MRI', 'EULER']):

    data = pd.read_table(FILES['clinical'])

    data.drop(columns=drop_cols, inplace=True)

    if z_score:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        with open('clinical_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        data = data.values

    path = os.path.join(datasetdir, 'clinical.npy')
    np.save(path, data)

    return path

def apply_qc(data, prefix, qc):
    idx_to_keep = pd.Series([True] * len(data))

    relation_mapper = {
        'gt': >, 'lt': <, 'gte': >=, 'lte': <=, 'eq': ==
    }
    for name, controls in qc.items():
        for relation, value in controls.items():
            if relation not in relation_mapper.keys():
                raise ValueError(f'The relationship {relation} provided is not a valid one')
            else:
                op = relation_mapper[relation]
                idx_to_keep = idx_to_keep & ( data[f"{prefix}-{name}"] op value )
    return data[idx_to_keep]

def fetch_rois(datasetdir=SAVING_FOLDER,
    metrics=['lgi:avg', 'thick:avg', 'surf:area'],
    roi_types=['cortical'], z_score=True,
    adjust_sites=True, residualize_by=['age', 'wisc:fsiq', 'sex'],
    qc={'wsic:fsiq': {'gte': 70}, 'euler': {'gt': -217}, 'mri': {'eq': 1}},
    test_size=0.2, seed=42):

    clinical_prefix = 'bloc-clinical_scores'

    roi_prefix = 'bloc-t1w_roi'
    
    data = pd.read_table(FILES['clinical_rois'])
    roi_mapper pd.read_table(FILES['roi_mapper'])

    roi_label_range = pd.Series([False] * len(roi_mapper))
    for roi_type in roi_types:
        if roi_type == 'cortical':
            roi_label_range = roi_label_range | ( 
                (roi_mapper['labels'] > 11000) & (roi_mapper['labels'] < 13000) )
        elif roi_type == 'subcortical':
            roi_label_range = roi_label_range | ( roi_mapper['labels'] > 13000 )
        elif roi_type == 'other':
            roi_label_range = roi_label_range | ( roi_mapper['labels'] < 11000 )
        else:
            raise ValueError("Roi types must be either 'cortical', 'subcortical' or 'other'")
    
    roi_labels = roi_mapper.loc[[roi_label_range, 'labels']

    features_list = []
    for column in data.columns:
        roi = int(column.split(':')[1].split('_')[0])
        metric = column.split('-')[-1]
        if roi in roi_labels and metric in metrics:
            features_list.append(column)

    data = apply_qc(data, clinical_prefix, qc).sort("participants_id")

    X_train = data[feature_list].copy()

    if test_size != 0:
        X_train, X_test = train_test_split(X_train, test_size=test_size, random_state=seed)

    if z_score:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        with open('rois_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        if test_size != 0:
            X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        if test_size != 0:
            X_test = X_test.values

    path = os.path.join(datasetdir, 'HBN_rois_X_train.npy')
    np.save(path, data)

    if test_size != 0:
        path_test = os.path.join(datasetdir, 'HBN_rois_X_test.npy')
        np.save(path_test, data)
        return path, path_test

    return path

def fetch_surface(datasetdir=SAVING_FOLDER, z_score=True,
    drop_cols=['EID', 'Study Site', 'Age', 'Sex', 'WISC_FSIQ', 'MRI', 'EULER']):
    
    data = pd.read_table(FILES['clinical_surface'])

    data.drop(columns=drop_cols, inplace=True)

    if z_score:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        with open('clinical_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        data = data.values

    path = os.path.join(datasetdir, 'clinical.npy')
    np.save(path, data)

    return path







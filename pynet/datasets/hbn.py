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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.linear_model import LinearRegression
from neurocombat_sklearn import CombatModel as fortin_combat
from pynet.datasets.euaims import WRAPPERS

logger = logging.getLogger("pynet")

# Global parameters
COHORT_NAME = "HBN"

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

DEFAULTS = {
    "clinical": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True,
        "drop_cols": ["study site", "age", "sex", "wisc:fsiq", "mri", "euler"],
        "qc": {"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217},
               "mri": {"eq": 1}},
    },
    "rois": {
        "metrics": ["lgi:avg", "thick:avg", "surf:area"],
        "roi_types": ["cortical"], "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True, "adjust_sites": True,
        "residualize_by": {"continuous": ["age", "wisc:fsiq"],
                           "discrete": ["sex"]},
        "qc": {"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217},
               "mri": {"eq": 1}},
    },
    "surface": {
        "metrics": ["pial_lgi", "thickness"],
        "test_size": 0.2, "seed": 42, "return_data": False,
        "z_score": True, "adjust_sites": True,
        "residualize_by": {"continuous": ["age", "wisc:fsiq"],
                           "discrete": ["sex"]},
        "qc": {"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217},
               "mri": {"eq": 1}},
    },
    "multiblock": {
        "blocks": ['clinical', 'surface-lh', 'surface-rh'],
        "test_size": 0.2, "seed": 42,
        "qc": {"wisc:fsiq": {"gte": 70}, "euler": {"gt": -217},
               "mri": {"eq": 1}},
    }
}


def make_fetchers(datasetdir=SAVING_FOLDER):

    return {
        'clinical': WRAPPERS['clinical'](datasetdir=datasetdir, files=FILES,
                                         cohort=COHORT_NAME,
                                         subject_column_name="EID",
                                         defaults=DEFAULTS['clinical']),
        'rois': WRAPPERS['rois'](datasetdir=datasetdir, files=FILES,
                                 cohort=COHORT_NAME,
                                 site_column_name="study site",
                                 defaults=DEFAULTS['rois']),
        'surface-rh': WRAPPERS['surface'](hemisphere='rh', files=FILES,
                                          datasetdir=datasetdir,
                                          cohort=COHORT_NAME,
                                          site_column_name="study site",
                                          defaults=DEFAULTS['surface']),
        'surface-lh': WRAPPERS['surface'](hemisphere='lh', files=FILES,
                                          datasetdir=datasetdir,
                                          cohort=COHORT_NAME,
                                          site_column_name="study site",
                                          defaults=DEFAULTS['surface']),
    }


fetch_multiblock = WRAPPERS["multiblock"](
    datasetdir='/neurospin/tmp/CA263211/preprocessed_data/HBN', files=FILES,
    cohort=COHORT_NAME, subject_column_name="EID",
    defaults=DEFAULTS["multiblock"], make_fetchers_func=make_fetchers)

for name, fetcher in make_fetchers('/neurospin/tmp/CA263211/preprocessed_data/HBN').items():
    print(name)
    ret, subj = fetcher(test_size=0, return_data=True)
    print(ret.shape, subj.shape)
# item = fetch_multiblock()
# print(item)

# -*- coding: utf-8 -*-
########################################################################
# NSAp - Copyright (C) CEA, 2021
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
import time
import logging
import numpy as np
from pynet.datasets.euaims import WRAPPERS

logger = logging.getLogger("pynet")

# Global parameters
COHORT_NAME = "HBN"
FOLDER = "/neurospin/brainomics/2020_deepint/data"
SAVING_FOLDER = "/tmp/HBN"
FILES = {
    "stratification": os.path.join(FOLDER, "HBN_stratification.tsv"),
    "rois_mapper": os.path.join(FOLDER, "HBN_rois.tsv"),
    "surf_stratification": os.path.join(FOLDER, "HBN_surf_stratification.tsv"),
}
DEFAULTS = {
    "clinical": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True,
        "drop_cols": ["study site", "age", "sex", "wisc:fsiq", "mri",
                      "euler", "labels", "subgroups"],
        "qc": {"bloc-clinical_score-wisc:fsiq": {"gte": 70},
               "bloc-clinical_score-euler": {"gt": -217},
               "bloc-clinical_score-mri": {"eq": 1}}
    },
    "rois": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True, "adjust_sites": True,
        "metrics": ["lgi:avg", "thick:avg", "surf:area"],
        "roi_types": ["cortical"],
        "residualize_by": {"continuous": ["age", "wisc:fsiq"],
                           "discrete": ["sex"]},
        "qc": {"wisc:fsiq": {"gte": 70},
               "euler": {"gt": -217},
               "mri": {"eq": 1}}
    },
    "surface": {
        "test_size": 0.2, "seed": 42,
        "return_data": False, "z_score": True, "adjust_sites": True,
        "metrics": ["pial_lgi", "thickness"],
        "residualize_by": {"continuous": ["bloc-clinical_score-age",
                                          "bloc-clinical_score-wisc:fsiq"],
                           "discrete": ["bloc-clinical_score-sex"]},
        "qc": {"bloc-clinical_score-wisc:fsiq": {"gte": 70},
               "bloc-clinical_score-euler": {"gt": -217},
               "bloc-clinical_score-mri": {"eq": 1}}
    },
    "multiblock": {
        "test_size": 0.2, "seed": 42,
        "blocks": ["clinical", "surface-lh", "surface-rh"],
        "qc": {"bloc-clinical_score-wisc:fsiq": {"gte": 70},
               "bloc-clinical_score-euler": {"gt": -217},
               "bloc-clinical_score-mri": {"eq": 1}}
    }
}


def make_fetchers(datasetdir=SAVING_FOLDER):

    return {
        "clinical": WRAPPERS["clinical"](
            datasetdir=datasetdir, files=FILES, cohort=COHORT_NAME,
            defaults=DEFAULTS["clinical"]),
        "rois": WRAPPERS["rois"](
            datasetdir=datasetdir, files=FILES, cohort=COHORT_NAME,
            site_column_name="study site", defaults=DEFAULTS["rois"]),
        "surface-rh": WRAPPERS["surface"](
            hemisphere="rh", files=FILES, datasetdir=datasetdir,
            cohort=COHORT_NAME, defaults=DEFAULTS["surface"],
            site_column_name="study site"),
        "surface-lh": WRAPPERS["surface"](
            hemisphere="lh", files=FILES, datasetdir=datasetdir,
            cohort=COHORT_NAME, defaults=DEFAULTS["surface"],
            site_column_name="study site")
    }


def fetch_multiblock_hbn(datasetdir=SAVING_FOLDER, fetchers=make_fetchers,
                         surface=False):
    if surface:
        DEFAULTS["multiblock"]["blocks"] = ["clinical", "surface-lh",
                                            "surface-rh"]
    else:
        DEFAULTS["multiblock"]["blocks"] = ["clinical", "rois"]
    return WRAPPERS["multiblock"](
        datasetdir=datasetdir, files=FILES, cohort=COHORT_NAME,
        subject_column_name="participant_id", defaults=DEFAULTS["multiblock"],
        make_fetchers_func=make_fetchers)()

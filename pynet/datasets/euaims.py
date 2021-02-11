# -*- coding: utf-8 -*-
########################################################################
# NSAp - Copyright (C) CEA, 2019
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
import urllib
import shutil
import requests
import logging
import numpy as np
from collections import namedtuple
import pandas as pd
import sklearn
from pynet.datasets import Fetchers


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

FOLDER = "/neurospin/brainomics/2020_deepint/data"

FILES = {
    'clinical': os.path.join(FOLDER, 'EUAIMS_clinical.tsv'),
    'rois': os.path.join(FOLDER, 'EUAIMS_rois.tsv'),
    'clinical_rois': os.path.join(FOLDER, 'EUAIMS_stratification.tsv'),
    'subgroups_full': os.path.join(FOLDER, 'EUAIMS_subgroups.tsv'),
    'clinical_genetic_surface': os.path.join(FOLDER, 'EUAIMS_surf_stratification.tsv'),
    'subgroups': os.path.join(FOLDER, 'EUAIMS_subgroups_angeline.csv'),
    'subgroups_rbs': os.path.join(FOLDER, 'EUAIMS_subgroups_with_rbs_angeline.csv'),

}

logger = logging.getLogger("pynet")

tables = []
print("table read")
tables.append(pd.read_table(FILES[0]))
print("table read")
tables.append(pd.read_table(FILES[1]))
print("table read")
tables.append(pd.read_table(FILES[2]))
print("table read")
tables.append(pd.read_table(FILES[3]))
print("table read")
tables.append(pd.read_table(FILES[4]))
print("table read")
tables.append(pd.read_csv(FILES[5]))
print("table read")
tables.append(pd.read_csv(FILES[6]))

for table in tables:
    print(table.shape)
    print(table.describe())
    print(table.columns)

def fetch_clinical():
    table = pd.read_table
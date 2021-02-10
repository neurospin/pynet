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

FILES = [
    os.path.join(FOLDER, 'EUAIMS_clinical.tsv'),
    os.path.join(FOLDER, 'EUAIMS_rois.tsv'),
    os.path.join(FOLDER, 'EUAIMS_stratification.tsv'),
    os.path.join(FOLDER, 'EUAIMS_subgroups.tsv'),
    os.path.join(FOLDER, 'EUAIMS_surf_stratification.tsv'),
    os.path.join(FOLDER, 'EUAIMS_subgroups_angeline.csv'),
]

logger = logging.getLogger("pynet")
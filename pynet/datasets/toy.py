# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare registration dataset.
"""

# Imports
import os
import json
import h5py
import glob
import urllib
import shutil
import requests
import logging
import numpy as np
import scipy
import zipfile
import skimage
from collections import namedtuple
import pandas as pd
from pynet.datasets import Fetchers


# Global parameters
Item = namedtuple("Item", ["t1w_path", "t2w_path", "flair_path",
                           "template_path"])
ATLAS_URL = ("https://raw.github.com/nilearn/nilearn/master/nilearn/datasets/"
             "data/avg152T1_brain.nii.gz")
T1W_URL = ("https://raw.github.com/muschellij2/open_ms_data/master/"
           "cross_sectional/raw/patient01/T1W.nii.gz")
T2W_URL = ("https://raw.github.com/muschellij2/open_ms_data/master/"
           "cross_sectional/raw/patient01/T2W.nii.gz")
FLAIR_URL = ("https://raw.github.com/muschellij2/open_ms_data/master/"
             "cross_sectional/raw/patient01/FLAIR.nii.gz")
logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_toy(datasetdir):
    """ Fetch a toy dataset composed of Nifti images.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.

    Returns
    -------
    item: namedtuple
        a named tuple containing 't1w_path', 't2w_path', 'flair_path',
        and 'template_path'.
    """
    logger.info("Loading toy dataset...")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    kwargs = {}
    for name, url in (("t1w_path", T1W_URL), ("t2w_path", T2W_URL),
                      ("flair_path", FLAIR_URL), ("template_path", ATLAS_URL)):
        basename = url.split("/")[-1]
        path = os.path.join(datasetdir, basename)
        if not os.path.isfile(path):
            response = requests.get(url, stream=True)
            with open(path, "wb") as out_file:
                response.raw.decode_content = False
                shutil.copyfileobj(response.raw, out_file)
            del response
        kwargs[name] = path
    return Item(**kwargs)

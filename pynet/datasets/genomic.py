# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare geomic dataset.
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
from pynet.datasets import Fetchers


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])
URLS = [
    ("https://raw.githubusercontent.com/miguelperezenciso/DLpipeline/master/"
     "DATA/wheat.X"),
    ("https://raw.githubusercontent.com/miguelperezenciso/DLpipeline/master/"
     "DATA/wheat.Y"),
]
logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_genomic_pred(datasetdir, to_categorical=False):
    """ Fetch/prepare the genomic prediction dataset for pynet.

    Matrix Y contains the average grain yield, column 1: Grain yield for
    environment 1 and so on.
    Matrix X contains marker genotypes.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    to_categorical: bool, default False
        if set convert the observation to categories.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading genomic prediction dataset.")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_genomic_pred.tsv")
    desc_categorical_path = os.path.join(
        datasetdir, "pynet_genomic_categorical_pred.tsv")
    input_path = os.path.join(datasetdir, "pynet_genomic_pred_inputs.npy")
    if not os.path.isfile(desc_path):
        for cnt, url in enumerate(URLS):
            logger.debug("Processing {0}...".format(url))
            basename = url.split(os.sep)[-1]
            datafile = os.path.join(datasetdir, basename)
            if not os.path.isfile(datafile):
                response = requests.get(url, stream=True)
                with open(datafile, "wt") as out_file:
                    out_file.write(response.text)
                del response
            else:
                logger.debug(
                    "Data '{0}' already downloaded.".format(datafile))
        data_x = pd.read_csv(
            os.path.join(datasetdir, "wheat.X"), header=None, sep="\s+")
        data_y = pd.read_csv(
            os.path.join(datasetdir, "wheat.Y"), header=None, sep="\s+")
        logger.debug("Data X: {0}".format(data_x.shape))
        logger.debug("Data Y: {0}".format(data_y.shape))
        np.save(input_path, data_x.values.astype(float))
        data_y = data_y.values.T
        metadata = dict(("env{0}".format(idx), val)
                        for idx, val in enumerate(data_y))
        df = pd.DataFrame.from_dict(metadata)
        data_y_cat = [
            (np.round(val - np.min(val)) / 2).astype(int) for val in data_y]
        metadata = {}
        for idx, env in enumerate(data_y_cat):
            metadata["env{0}".format(idx)] = env
            labels = np.unique(env)
            env_cat = pd.get_dummies(env).values
            for key, val in zip(labels, env_cat.T):
                metadata["env{0}_cat{1}".format(idx, key)] = val
        cat_df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
        cat_df.to_csv(desc_categorical_path, sep="\t", index=False)
    desc_path = desc_categorical_path if to_categorical else desc_path
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=None)

# -*- coding: utf-8 -*-
########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
########################################################################

"""
Module provides functions to prepare different toy datasets from UKB.
  1) toy example about height in UKB with no NaN and known signif snps
  2)
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
import numpy as np
import sklearn
from pynet.datasets import Fetchers
from pandas_plink import read_plink
import matplotlib.pyplot as plt
import progressbar
import statsmodels.api as sm
import warnings


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

MSG = (
    "See https://gitlab.com/brainomics/brainomics_notebooks "
    "and the notebook "
    "notebooks/studies/HEIGHT_UKB_DeepLearning.ipynb"
)

logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_nicodep(file_name='nicodep_nd_aa',
    datasetdir='/neurospin/brainomics/2020_corentin_smoking/',
    visualize_labels=False, treat_nans='remove'):
    """ Fetch/prepare nicotine dependence dataset for pynet.

    Matrix Y contains the phenotypes
    Matrix X contains marker genotypes.

    Parameters
    ----------
    file_name: str, default 'nicodep_nd_aa'
        prefix of the files containing the data
    datasetdir: str
        the dataset destination folder.
    visualize_labels: bool, default False
        if set show a few histograms to see labels
        distribution over the dataset

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading nicotine dependence dataset.")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_{}_labels.tsv".format(file_name))
    input_path = os.path.join(datasetdir, "pynet_{}_inputs.npy".format(file_name))

    if not os.path.isfile(desc_path) or not os.path.isfile(desc_path):
        bim, fam, bed = read_plink(os.path.join(datasetdir, file_name))

        data_x = np.transpose(bed.compute())

        logger.info("Data X: {0}".format(data_x.shape))
        # Get data_y
        data_y = pd.read_csv(
            os.path.join(datasetdir, "nicodep.pheno"), sep=" ")
        data_y = data_y.astype({
            'FID':str,
            'IID':str,
            }).set_index(['FID', 'IID'])
        data_y.drop('gender', axis=1, inplace=True)

        data_y = fam.join(data_y, on=['fid', 'iid'])
        data_y.reset_index(inplace=True)
        data_y.drop(['fid', 'iid', 'tissue', 'mother', 'father', 'ethnicity', 'gender', 'trait', 'i', 'index'], axis=1, inplace=True)
        data_y = data_y - 1
        data_y = data_y.astype({'smoker': int})

        if visualize_labels:
            for label in data_y.columns.tolist()[2:]:
                data_y.hist(label, bins = len(data_y[label].unique()))
            plt.show()

        data_y.replace(-9, -1, inplace=True)
        logger.info("Data Y: {0}".format(data_y.shape))

        categorical_pheno = []
        for pheno in categorical_pheno:
            dummy_values = pd.get_dummies(data_y[pheno], prefix="{0}_cat".format(pheno))
            data_y = pd.concat([data_y, dummy_values], axis=1)

        if treat_nans == 'remove':
            data_x = data_x[:, ~np.isnan(data_x.sum(axis=0))]

        np.save(input_path, data_x.astype(float))
        data_y.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=None)

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
import sklearn
from pynet.datasets import Fetchers
from pandas_plink import read_plink


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
def fetch_aa_nicodep(datasetdir='/neurospin/brainomics/2020_corentin_smoking/', to_categorical=False, check=False):
    """ Fetch/prepare nicotine dependence dataset for pynet.

    Matrix Y contains the average grain yield, column 1: Grain yield for
    environment 1 and so on.
    Matrix X contains marker genotypes.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    to_categorical: bool, default False
        if set convert the observation to categories.
    check: bool, default False
        if set check results against the downloaded .check file data

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading nicotine dependence dataset.")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_aa_nicodep_labels.tsv")
    desc_categorical_path = os.path.join(
        datasetdir, "pynet_aa_nicodep_categorical_labels.tsv")
    input_path = os.path.join(datasetdir, "pynet_aa_nicodep_inputs.npy")
    #file_todel = []
    if not os.path.isfile(desc_path):
        bim, fam, bed = read_plink(os.path.join(datasetdir, 'nicodep_aa'))

        data_x = bed.compute()

        logger.info("Data X: {0}".format(data_x.shape))

        # Get data_y
        #  Cosmetics
        data_y = pd.read_csv(
            os.path.join(datasetdir, "nicodep.pheno"), sep=" ")
        # data_y = pd.read_csv(
        #     os.path.join(datasetdir, "toy_height.phe"), sep="\t")
        print(data_y.head())
        print(fam.head())
        print(bim.head())
        
        # data_y.drop(['FID', 'IID'], axis=1, inplace=True)
        # cov.drop(['FID', 'IID'], axis=1, inplace=True)
        logger.info("Data Y: {0}".format(data_y.shape))

    #      residualize
    #     logger.info("Residualize Data Y")
    #     import statsmodels.api as sm
    #     y = data_y.values
    #     X = cov.values
    #     X = sm.add_constant(X)
    #     model = sm.OLS(y, X, missing='drop')
    #     results = model.fit()
    #     y_res = y - results.predict(X).reshape(-1, 1)
    #     data_y['Height'] = y_res
    #     data_y['HeightCat'] = pd.qcut(data_y.Height, q=3, labels=[1, 2, 3])
    #     tmpdf = pd.get_dummies(data_y.HeightCat)
    #     d = {}
    #     for i in tmpdf.columns:
    #         d[i] = "Height_{}".format(i)
    #     tmpdf.rename(d, axis='columns', inplace=True)
    #     data_y = pd.concat([data_y, tmpdf], axis=1)

    #     # now data_y colomns are Height, HeightCat, HeigthCat_0, ..
    #     maskcolumns = data_y.columns.tolist()
    #     maskcolumns.remove('Height')
    #     logger.info("Save Data Y")
    #     data_y[['Height']].to_csv(desc_path, sep="\t", index=False)
    #     logger.info("Save Data Y (categorical)")
    #     data_y[maskcolumns].to_csv(desc_categorical_path,
    #                                sep="\t", index=False)
    #     logger.info("Save Data X")
    #     np.save(input_path, data_x.astype(float))

    #     # Housekeeping
    #     desc_path = desc_categorical_path if to_categorical else desc_path
    #     for f in file_todel:
    #         os.remove(f)
    # return Item(input_path=input_path, output_path=None,
    #             metadata_path=desc_path, labels=None)

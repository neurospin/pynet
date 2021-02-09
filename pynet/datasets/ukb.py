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


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

FILES = [
    ("/neurospin/ukb/derivatives/brainomics_multivariate/"
     "toy_height.phe"),
    ("/neurospin/ukb/derivatives/brainomics_multivariate/"
     "toy_age_sex.cov"),
    ("/neurospin/ukb/derivatives/brainomics_multivariate/"
     "toy_chr19_chunk7_nonan.npz"),
    ("/neurospin/ukb/derivatives/brainomics_multivariate/"
     "toy_chr19_chunk7_nonan.check"),
]
MSG = (
    "See https://gitlab.com/brainomics/brainomics_notebooks "
    "and the notebook "
    "notebooks/studies/HEIGHT_UKB_DeepLearning.ipynb"
)

logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_height_biobank(datasetdir, to_categorical=False, check=False):
    """ Fetch/prepare the height biobank prediction dataset for pynet.

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
    logger.info("Loading UK BioBank height dataset.")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_bb_height_pred.tsv")
    desc_categorical_path = os.path.join(
        datasetdir, "pynet_bb_height_categorical_pred.tsv")
    input_path = os.path.join(datasetdir, "pynet_bb_height_pred_inputs.npy")
    file_todel = []
    if not os.path.isfile(desc_path):
        for cnt, fname in enumerate(FILES):
            logger.debug("Processing {0}...".format(fname))
            basename = fname.split(os.sep)[-1]
            datafile = os.path.join(datasetdir, basename)
            if not os.path.isfile(datafile):
                shutil.copy(fname, datafile)
                file_todel.append(datafile)
            else:
                logger.debug(
                    "Data '{0}' already downloaded.".format(datafile))

        # Get data_x, read from the chunk #7 nan filteredout
        data_x = np.load(os.path.join(datasetdir,
                                      "toy_chr19_chunk7_nonan.npz"),
                         allow_pickle=True
                         )['genotype']
        logger.info("Data X: {0}".format(data_x.shape))

        # Get data_y
        #  Cosmetics
        cov = pd.read_csv(
            os.path.join(datasetdir, "toy_age_sex.cov"), sep="\t")
        data_y = pd.read_csv(
            os.path.join(datasetdir, "toy_height.phe"), sep="\t")
        data_y.drop(['FID', 'IID'], axis=1, inplace=True)
        cov.drop(['FID', 'IID'], axis=1, inplace=True)
        logger.info("Data Y: {0}".format(data_y.shape))

        #  residualize
        logger.info("Residualize Data Y")
        import statsmodels.api as sm
        y = data_y.values
        X = cov.values
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing='drop')
        results = model.fit()
        y_res = y - results.predict(X).reshape(-1, 1)
        data_y['Height'] = y_res
        data_y['HeightCat'] = pd.qcut(data_y.Height, q=3, labels=[1, 2, 3])
        tmpdf = pd.get_dummies(data_y.HeightCat)
        d = {}
        for i in tmpdf.columns:
            d[i] = "Height_{}".format(i)
        tmpdf.rename(d, axis='columns', inplace=True)
        data_y = pd.concat([data_y, tmpdf], axis=1)

        if check:
            # check data coherence
            # check shapes
            assert (data_x.shape[0] == cov.shape[0])
            assert (data_x.shape[0] == data_y.shape[0])
            # check unvariate SNP p wrt check file
            pvals_res = []
            for idx in range(data_x.shape[1]):
                y = y_res
                X = data_x[:, idx].reshape(-1, 1)
                X = sm.add_constant(X)
                model = sm.OLS(y, X, missing='drop')
                results_res = model.fit()
                pvals_res.append((results_res.pvalues[0]))
            pvals_res = np.array(pvals_res)
            ref = pd.read_csv(
                          os.path.join(datasetdir,
                                       "toy_chr19_chunk7_nonan.check"),
                          sep="\t")
            ref['runtimeP'] = pvals_res
            ref.sort_values('P from residual').head(20)
            np.testing.assert_almost_equal(
                                     ref['runtimeP'].tolist(),
                                     ref['P from residual'].tolist()
                                     )
        # now data_y colomns are Height, HeightCat, HeigthCat_0, ..
        maskcolumns = data_y.columns.tolist()
        maskcolumns.remove('Height')
        logger.info("Save Data Y")
        data_y[['Height']].to_csv(desc_path, sep="\t", index=False)
        logger.info("Save Data Y (categorical)")
        data_y[maskcolumns].to_csv(desc_categorical_path,
                                   sep="\t", index=False)
        logger.info("Save Data X")
        np.save(input_path, data_x.astype(float))

        # Housekeeping
        desc_path = desc_categorical_path if to_categorical else desc_path
        for f in file_todel:
            os.remove(f)
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=None)

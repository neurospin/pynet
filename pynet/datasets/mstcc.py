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
def fetch_aa_nicodep(datasetdir='/neurospin/brainomics/2020_corentin_smoking/',
    visualize_labels=False, treat_nans='remove')#p_value_filter=None, N_best=None):
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
    input_path = os.path.join(datasetdir, "pynet_aa_nicodep_inputs.npy")
    #file_todel = []
    if not os.path.isfile(desc_path) or not os.path.isfile(desc_path):
        bim, fam, bed = read_plink(os.path.join(datasetdir, 'nicodep_aa'))

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
        data_y.drop(['fid', 'iid', 'smoking_status', 'father', 'mother', 'tissue', 'i', 'ethnicity'], axis=1, inplace=True)
        data_y.set_index('index', inplace=True)
        data_y = data_y.astype({'trait': int})
        data_y.rename(columns={'trait':'smoker'}, inplace=True)
        data_y['smoker'].replace([2, 1], [1, 0], inplace=True)

        if visualize_labels:
            for label in data_y.columns.tolist()[2:]:
                data_y.hist(label, bins = len(data_y[label].unique()))
            plt.show()

        data_y.replace(-9, np.nan, inplace=True)
        logger.info("Data Y: {0}".format(data_y.shape))

        categorical_pheno = []
        for pheno in categorical_pheno:
            dummy_values = pd.get_dummies(data_y[pheno], prefix="{0}_cat".format(pheno))
            data_y = pd.concat([data_y, dummy_values], axis=1)

        if treat_nans == 'remove':
            data_x = data_x[:, ~np.isnan(data_x.sum(axis=0))]

        # if p_value_filter or N_best:

        #     pvals = []
        #     data_x = data_x[:, ~np.isnan(data_x.sum(axis=0))]

        #     pbar = progressbar.ProgressBar(
        #         max_value=data_x.shape[1], redirect_stdout=True, prefix="Filtering snps ")

        #     n_errors = 0
        #     pbar.start()
        #     for idx in range(data_x.shape[1]):
        #         pbar.update(idx+1)
        #         X = np.concatenate([
        #             data_x[:, idx, np.newaxis],
        #             data_y[['age']].values,
        #             data_y[['gender']].astype(int).values], axis=1)
        #         X = sm.add_constant(X)

        #         model = sm.Logit(data_y['smoker'].values, X, missing='drop')

        #         with warnings.catch_warnings():
        #             warnings.filterwarnings("ignore")
        #             try:
        #                 results_res = model.fit(disp=0)
        #                 pvals.append((results_res.pvalues[0]))
        #             except:
        #                 pvals.append(1)
        #                 n_errors += 1
        #     #print(n_errors)
        #     pvals = np.array(pvals)

        #     if N_best:
        #         snp_list = pvals.argsort()[:N_best].squeeze().tolist()

        #     if p_value_filter:
        #         snp_list_tmp = np.nonzero(pvals < p_value_filter)[0].squeeze().tolist()
        #         if N_best:
        #             snp_list = list(set(snp_list).intersection(snp_list_tmp))
        #         else:
        #             snp_list = snp_list_tmp

        #     data_x = data_x[:, snp_list]
        #     pbar.finish()

        np.save(input_path, data_x.astype(float))
        data_y.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=None)

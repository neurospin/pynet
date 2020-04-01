# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the IMPAC dataset. IMPAC stands for
IMaging-PsychiAtry Challenge: predicting autism which is a data challenge on
Autism Spectrum Disorder detection:
https://paris-saclay-cds.github.io/autism_challenge.
"""

# Imports
import re
import os
import json
import glob
import shutil
import logging
import requests
import zipfile
import hashlib
import warnings
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from pynet.datasets import Fetchers
try:
    from nilearn.connectome import ConnectivityMeasure
except:
    warnings.warn("You need to install nilearn.")


# Global parameters
ATLAS = ("basc064", "basc122", "basc197", "craddock_scorr_mean",
         "harvard_oxford_cort_prob_2mm", "msdl", "power_2011")
ARCHIVE = {
    atlas: 'https://zenodo.org/record/3625740/files/{0}.zip'.format(atlas)
    for atlas in ATLAS}
CHECKSUM = {
    "basc064":
    "75eb5ee72344d11f056551310a470d00227fac3e87b7205196f77042fcd434d0",
    "basc122":
    "2d0d2c2338f9114877a0a1eb695e73f04fc664065d1fb75cff8d59f6516b0ec7",
    "basc197":
    "68135bb8e89b5b3653e843745d8e5d0e92876a5536654eaeb9729c9a52ab00e9",
    "craddock_scorr_mean":
    "634e0bb07beaae033a0f1615aa885ba4cb67788d4a6e472fd432a1226e01b49b",
    "harvard_oxford_cort_prob_2mm":
    "638559dc4c7de25575edc02e58404c3f2600556239888cbd2e5887316def0e74",
    "msdl":
    "fd241bd66183d5fc7bdf9a115d7aeb9a5fecff5801cd15a4e5aed72612916a97",
    "power_2011":
    "d1e3cd8eaa867079fe6b24dfaee08bd3b2d9e0ebbd806a2a982db5407328990a"}
URL = "https://raw.githubusercontent.com/ramp-kits/autism/master/data/"
URLS = [URL + name for name in ["anatomy.csv", "anatomy_qc.csv",
                                "fmri_filename.csv", "fmri_qc.csv",
                                "fmri_repetition_time.csv",
                                "participants.csv", "test.csv", "train.csv"]]
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels", "nb_features"])
logger = logging.getLogger("pynet")


def _sha256(path):
    """ Calculate the sha256 hash of the file at path.
    """
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _check_and_unzip(zip_file, atlas, atlas_directory):
    checksum_download = _sha256(zip_file)
    if checksum_download != CHECKSUM[atlas]:
        os.remove(zip_file)
        raise IOError("The file downloaded was corrupted. Try again "
                      "to execute this fetcher.")
    logger.info("Decompressing the archive...")
    zip_ref = zipfile.ZipFile(zip_file, "r")
    zip_ref.extractall(atlas_directory)
    zip_ref.close()


def _download_fmri_data(atlas, outdir):
    logger.info("Downloading the data from {0}...".format(ARCHIVE[atlas]))
    zip_file = os.path.join(outdir, atlas + ".zip")
    if os.path.isfile(zip_file):
        logger.info("'{0}' already downloaded!".format(zip_file))
    else:
        response = requests.get(ARCHIVE[atlas])
        with open(zip_file, "wb") as of:
            of.write(response.content)
    atlas_directory = os.path.join(outdir, "data", "fmri")
    if not os.path.isdir(atlas_directory):
        _check_and_unzip(zip_file, atlas, atlas_directory)


def fetch_fmri_time_series(outdir, atlas="all"):
    """ Fetch the time-series extracted from the fMRI data using a specific
    atlas.

    Parameters
    ----------
    outdir: str
        the detination folder.
    atlas : string, default='all'
        The name of the atlas used during the extraction. The possibilities
        are:
        * `'basc064`, `'basc122'`, `'basc197'`: BASC parcellations with 64,
        122, and 197 regions [1]_;
        * `'craddock_scorr_mean'`: Ncuts parcellations [2]_;
        * `'harvard_oxford_cort_prob_2mm'`: Harvard-Oxford anatomical
        parcellations;
        * `'msdl'`: MSDL functional atlas [3]_;
        * `'power_2011'`: Power atlas [4]_.

    References
    ----------
    .. [1] Bellec, Pierre, et al. "Multi-level bootstrap analysis of stable
       clusters in resting-state fMRI." Neuroimage 51.3 (2010): 1126-1139.
    .. [2] Craddock, R. Cameron, et al. "A whole brain fMRI atlas generated
       via spatially constrained spectral clustering." Human brain mapping
       33.8 (2012): 1914-1928.
    .. [3] Varoquaux, Gaël, et al. "Multi-subject dictionary learning to
       segment an atlas of brain spontaneous activity." Biennial International
       Conference on Information Processing in Medical Imaging. Springer,
       Berlin, Heidelberg, 2011.
    .. [4] Power, Jonathan D., et al. "Functional network organization of the
       human brain." Neuron 72.4 (2011): 665-678.
    """
    if atlas == "all":
        for single_atlas in ATLAS:
            _download_fmri_data(single_atlas, outdir)
    elif atlas in ATLAS:
        _download_fmri_data(atlas, outdir)
    else:
        raise ValueError(
            "'atlas' should be one of {0}. Got {1} instead.".format(
                ATLAS, atlas))
    logger.info("Downloading completed...")


def _load_fmri(fmri_filenames):
    """ Load time-series extracted from the fMRI using a specific atlas.
    """
    return np.array([pd.read_csv(subject_filename,
                                 header=None).values
                     for subject_filename in fmri_filenames])


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """ Make a transformer which will load the time series and compute the
    connectome matrix.
    """
    def __init__(self):
        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind="tangent", vectorize=True))

    def fit(self, X_df, y, datadir):
        fmri_filenames = [path.replace(".", datadir, 1)
                          for path in X_df["fmri_basc122"]]
        self.transformer_fmri.fit(fmri_filenames, y)
        return self

    def transform(self, X_df, datadir):
        fmri_filenames = [path.replace(".", datadir, 1)
                          for path in X_df["fmri_basc122"]]
        X_connectome = self.transformer_fmri.transform(fmri_filenames)
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ["connectome_{0}".format(i)
                                for i in range(X_connectome.columns.size)]
        X_anatomy = X_df[[col for col in X_df.columns
                          if col.startswith("anatomy")]]
        X_anatomy = X_anatomy.drop(columns="anatomy_select")
        logger.debug("  X connectome: {0}".format(X_connectome.shape))
        logger.debug("  X anatomy: {0}".format(X_anatomy.shape))
        return pd.concat([X_connectome, X_anatomy], axis=1)


@Fetchers.register
def fetch_impac(datasetdir, mode="train", dtype="all"):
    """ Fetch/prepare the IMPAC dataset for pynet.

    To compute the functional connectivity using the rfMRI data, we use the
    BASC atlas with 122 ROIs.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    mode: str
        ask the 'train' or 'test' dataset.
    dtype: str, default 'all'
        the features type: 'anatomy', 'fmri', or 'all'.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading impac dataset.")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    train_desc_path = os.path.join(datasetdir, "pynet_impac_train.tsv")
    selected_input_path = os.path.join(
            datasetdir, "pynet_impac_inputs_selection.npy")
    train_input_path = os.path.join(
        datasetdir, "pynet_impac_inputs_train.npy")
    train_output_path = os.path.join(
        datasetdir, "pynet_impac_outputs_train.npy")
    test_desc_path = os.path.join(datasetdir, "pynet_impac_test.tsv")
    test_input_path = os.path.join(
        datasetdir, "pynet_impac_inputs_test.npy")
    test_output_path = os.path.join(
        datasetdir, "pynet_impac_outputs_test.npy")
    if not os.path.isfile(train_desc_path):
        fetch_fmri_time_series(datasetdir, atlas="basc122")
        data = []
        sets = {}
        for url in URLS:
            basename = url.split("/")[-1]
            name = basename.split(".")[0]
            local_file = os.path.join(datasetdir, basename)
            if not os.path.isfile(local_file):
                response = requests.get(url, stream=True)
                with open(local_file, "wt") as out_file:
                    out_file.write(response.text)
                del response
            else:
                logger.info("'{0}' already downloaded!".format(basename))
            if name not in ("train", "test"):
                prefix = name.split("_")[0]
                df = pd.read_csv(local_file, index_col=0)
                df.columns = [
                    "{0}_{1}".format(prefix, col) for col in df.columns]
                data.append(df)
            else:
                df = pd.read_csv(local_file, header=None)
                sets[name] = df[0].values.tolist()
        data = pd.concat(data, axis=1)
        logger.debug("  data: {0}".format(data.shape))
        data = data[data["anatomy_select"].isin((1, 2))]
        data = data[data["fmri_select"].isin((1, 2))]
        logger.debug("  filter data: {0}".format(data.shape))
        data_train = data[data.index.isin(sets["train"])]
        data_test = data[data.index.isin(sets["test"])]
        y_train = data_train["participants_asd"]
        y_test = data_test["participants_asd"]
        logger.debug("  data train: {0}".format(data_train.shape))
        logger.debug("  data test: {0}".format(data_test.shape))
        logger.debug("  y train: {0}".format(y_train.shape))
        logger.debug("  y test: {0}".format(y_test.shape))
        features = FeatureExtractor()
        features.fit(data_train, y_train, datasetdir)
        features_train = features.transform(data_train, datasetdir)
        features.fit(data_test, y_test, datasetdir)
        features_test = features.transform(data_test, datasetdir)
        logger.debug("  features train: {0}".format(features_train.shape))
        logger.debug("  features test: {0}".format(features_test.shape))
        np.save(train_input_path, features_train)
        np.save(train_output_path, y_train)
        np.save(test_input_path, features_test.values)
        np.save(test_output_path, y_test.values)
        data_train.to_csv(train_desc_path, sep="\t", index=False)
        data_test.to_csv(test_desc_path, sep="\t", index=False)
    if mode == "train":
        input_path, output_path, desc_path = (
            train_input_path, train_output_path, train_desc_path)
    else:
        input_path, output_path, desc_path = (
            test_input_path, test_output_path, test_desc_path)
    features = np.load(input_path)
    if dtype == "anatomy":
        features = features[:, 7503:]
    elif dtype == "fmri":
        features = features[:, :7503]
    nb_features = features.shape[1]
    np.save(selected_input_path, features)
    return Item(input_path=selected_input_path, output_path=None,
                metadata_path=desc_path, labels=None, nb_features=nb_features)

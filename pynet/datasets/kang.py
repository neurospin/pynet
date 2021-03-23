# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the Kang dataset.

H. M. Kang, et al., Multiplexed droplet single-cell rna-sequencing
using natural genetic variation. Nature biotechnology, 2018.
"""

# Imports
import os
import json
import logging
import subprocess
import requests
from collections import namedtuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pynet.datasets import Fetchers


# Global parameters
Item = namedtuple("Item", ["data", "trainset", "testset", "membership_mask"])
logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_kang(datasetdir, random_state=None):
    """ Download the Kang dataset described in [1].

    [1] H. M. Kang, et al., Multiplexed droplet single-cell rna-sequencing
    using natural genetic variation.Nature biotechnology, 2018.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    random_state: int, default None
        controls the shuffling applied to the data before applying the split.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'data', 'trainset', 'testset' and
        'membership_mask'.
    """
    logger.info("Loading Kang dataset.")

    # Fisrt import specific modules
    try:
        import anndata
    except:
        raise ImportError("Please install anndata to use 'fetch_kang'.")

    # Download resources
    url_data = ("https://docs.google.com/uc?export=download&id=1-N7wPpYUf_"
                "QcG5566WVZlaxVC90M7NNE")
    url_gt = ("https://public.bmi.inf.ethz.ch/projects/2020/pmvae/"
              "kang_recons.h5ad")
    url_gmt = ("https://raw.githubusercontent.com/ratschlab/pmvae/main/data/"
               "c2.cp.reactome.v4.0.symbols.gmt")
    origdatapath = os.path.join(datasetdir, "orig_kang_count.h5ad")
    datapath = os.path.join(datasetdir, "kang_count.h5ad")
    gtpath = os.path.join(datasetdir, "kang_recons.h5ad")
    gmtpath = os.path.join(datasetdir, "c2.cp.reactome.v4.0.symbols.gmt")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    if not os.path.isfile(origdatapath):
        cmd = ["wget", "--no-check-certificate", url_data, "-O", origdatapath]
        subprocess.check_call(cmd)
    if not os.path.isfile(datapath):
        data = anndata.read(origdatapath)
        data.obs = data.obs[["condition", "cell_type"]]
        data.uns = dict()
        data.obsm = None
        data.varm = None
        data.write(datapath)
    if not os.path.isfile(gtpath):
        cmd = ["wget", url_gt, "-O", gtpath]
        subprocess.check_call(cmd)
    if not os.path.isfile(gmtpath):
        response = requests.get(url_gmt)
        with open(gmtpath, "wt") as open_file:
            open_file.write(response.text)

    # Build dataset
    data = anndata.read(datapath)
    data.varm["annotations"] = load_annotations(
        gmtpath, data.var_names, min_genes=13)
    membership_mask = data.varm["annotations"].astype(bool).T
    logger.info("-- membership mask: {0}".format(
        membership_mask.values.shape))
    trainset, testset = train_test_split(
        data.X, test_size=0.25, shuffle=True, random_state=random_state)
    logger.info("-- trainset: {0}".format(trainset.shape))
    logger.info("-- testset: {0}".format(testset.shape))

    return Item(data=data.X, trainset=trainset, testset=testset,
                membership_mask=membership_mask)


def load_annotations(gmt, genes, min_genes=10):
    genesets = parse_gmt(gmt, genes, min_genes)
    annotations = pd.DataFrame(False, index=genes, columns=genesets.keys())
    for key, genes in genesets.items():
        annotations.loc[genes, key] = True
    return annotations


def parse_gmt(path, symbols=None, min_genes=10):
    lut = dict()
    for line in open(path, "r"):
        key, _, *genes = line.strip().split()
        if symbols is not None:
            genes = symbols.intersection(genes).tolist()
        if len(genes) < min_genes:
            continue
        lut[key] = genes
    return lut

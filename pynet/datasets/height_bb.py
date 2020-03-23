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


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])
FILES = [
    ("/home/vf140245/data/external/"
     "ukb_height.phe"),
    ("/home/vf140245/data/external/"
     "ukb_age_sex.cov"),
    ("/home/vf140245/data/external/"
     "snp_val_7.npz"),
]
MSG = (
    "See https://gitlab.com/brainomics/brainomics_notebooks "
    "and the notebook height_DL_data_preparation.py"
)

logger = logging.getLogger("pynet")

"""
tmp = []
from glob import glob
for fn in glob('/home/vf140245/data/external/snp_name_*.npz'):
	print(fn)
	a = np.load(fn, allow_pickle=True)['arr_0']
	a = [i.split('_')[1] for i in a]
	tmp.append(a)
from swissknife import read_sumstat
df = read_sumstat('/home/vf140245/data/local/height_gwas19.assoc.linear')
df.set_index('SNP', inplace=True)
df['GENO'] = -1

for i, a in enumerate(tmp):
	asub = list(set(a).intersection(df.index.tolist()))
	df.loc[asub, 'GENO'] = i
df.loc[df.P < 1e-8].groupby('GENO').count()
      CHR  BP  A1  TEST  NMISS  BETA  STAT   P
GENO                                          
0      26  26  26    26     26    26    26  26
1       6   6   6     6      6     6     6   6
2      10  10  10    10     10    10    10  10
3      19  19  19    19     19    19    19  19
4      24  24  24    24     24    24    24  24
5      37  37  37    37     37    37    37  37
6       2   2   2     2      2     2     2   2
7      42  42  42    42     42    42    42  42
8      41  41  41    41     41    41    41  41
9      32  32  32    32     32    32    32  32

Choix du chunk #7
"""


def fetch_height_biobank(datasetdir, to_categorical=False):
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

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading UK BioBank height dataset.")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_genomic_bb_height.tsv")
    desc_categorical_path = os.path.join(
        datasetdir, "pynet_genomic_bb_height_categorical_pred.tsv")
    input_path = os.path.join(datasetdir, "pynet_genomic_bb_height_inputs.npy")
    if not os.path.isfile(desc_path):
        for cnt, fname in enumerate(FILES):
            logger.debug("Processing {0}...".format(fname))
            basename = fname.split(os.sep)[-1]
            datafile = os.path.join(datasetdir, basename)
            if not os.path.isfile(datafile):
                shutil.copy(fname, datafile)
            else:
                logger.debug(
                    "Data '{0}' already downloaded.".format(datafile))
        data_x = np.load(os.path.join(datasetdir, "snp_val_7.npz"),
                         allow_pickle=True
                         )['arr_0']
        data_y = pd.read_csv(
            os.path.join(datasetdir, "ukb_height.phe"), sep=",")
        logger.debug("Data X: {0}".format(data_x.shape))
        logger.debug("Data Y: {0}".format(data_y.shape))
        # ~ np.save(input_path, data_x.values.astype(float))
        data_y.drop(['FID', 'IID'], axis=1, inplace=True)
        data_y['HeightCat'] =  (np.round(data_y.Height - 120.)/15).astype(int)
        data_y.loc[data_y.HeightCat <0, 'HeightCat'] = 0
        tmpdf = pd.get_dummies(data_y.HeightCat)
        for i in tmpdf.columns:
			tmpdf.rename({int("{}".format(i)):"Height_{}".format(i)}, 
			axis='columns')
        data_y = pd.concat([data_y,tmpdf], axis=1)
	    # now data_y colomns are Height, HeightCat, HeigthCat_0, ..
	    maskcolumns = data_y.columns.tolist()
	    maskcolumns.remove('Height')
        data_y[['Height']].to_csv(desc_path, sep="\t", index=False)
        data_y[maskcolumns]to_csv(desc_categorical_path, sep="\t", index=False)
    desc_path = desc_categorical_path if to_categorical else desc_path
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=None)

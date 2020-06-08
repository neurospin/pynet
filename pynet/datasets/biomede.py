# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the Biomede dataset.
"""

# Imports
from collections import namedtuple
import os
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import progressbar
from pynet.datasets import Fetchers

# Global parameters
MODALITIES = ("t1", "t1ce", "t2", "flair")
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])
logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_biomede(datasetdir):
    """ Fetch/prepare the Biomede dataset for pynet.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading biomede dataset.")

    def _crop(arr):
        return arr[45: 195, 30: 220, :145]

    def _norm(arr, mask=None):
        logical_mask = (arr != 0) if mask is None else mask
        mean = arr[logical_mask].mean()
        std = arr[logical_mask].std()
        return ((arr - mean) / std).astype(np.single)

    traindir = os.path.join(datasetdir, "biomede")
    mapping_path = os.path.join(traindir, "name_mapping.csv")
    if not os.path.isfile(mapping_path):
        raise ValueError("This dataset is private. You need special access")
    desc_path = os.path.join(datasetdir, "pynet_biomede.tsv")
    input_path = os.path.join(datasetdir, "pynet_biomede_inputs.npy")
    output_path = os.path.join(datasetdir, "pynet_biomede_outputs.npy")
    if not os.path.isfile(desc_path):
        df = pd.read_csv(mapping_path, sep=",")
        input_dataset = []
        output_dataset = []
        nb_subjects = df.shape[0]
        with progressbar.ProgressBar(max_value=nb_subjects,
                                     redirect_stdout=True) as bar:
            for cnt, row in enumerate(df.iterrows()):
                logger.debug("Processing {0}_{1}...".format(row["sub"],
                                                            row["ses"]))
                data = []
                for mod in MODALITIES:
                    if (row[mod] is None):
                        continue
                    mask = np.where(nib.load(row["mask"]).get_fdata() > 0) \
                        if row["mask"] is not None else None
                    data.append(_norm(_crop(nib.load(row[mod]).get_data()),
                                      mask))
                data = np.asarray(data)
                input_dataset.append(data)
                bar.update(cnt)
            input_dataset = np.asarray(input_dataset)
            np.save(input_path, input_dataset)
            output_dataset = np.asarray(output_dataset)
            np.save(output_path, output_dataset)
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)

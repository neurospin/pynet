# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the metastasis dataset.
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
MODALITIES = ("t1", "flair")
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])
logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_metastasis(
        datasetdir="/neurospin/radiomics_pub/workspace/metastasis_dl/data"):
    """ Fetch/prepare the metastatis dataset for pynet.

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
    logger.info("Loading metastasis dataset.")

    def _crop(arr):
        # return arr[45: 195, 30: 220, 10: 145]
        return arr

    def _norm(arr):
        arr = np.nan_to_num(arr)
        logical_mask = (arr != 0)
        mean = arr[logical_mask].mean()
        std = arr[logical_mask].std()
        return ((arr - mean) / std).astype(np.single)

    mapping_path = os.path.join(datasetdir, "dataset_categorial.tsv")
    if not os.path.isfile(mapping_path):
        raise ValueError(
            "Are you in the right folder? "
            "You may need special access for Metastasis dataset")
    desc_path = os.path.join(datasetdir, "pynet_metastasis.tsv")
    input_path = os.path.join(datasetdir, "pynet_metastasis_inputs.npy")
    output_path = os.path.join(datasetdir, "pynet_metastasis_outputs.npy")
    if not os.path.isfile(desc_path):
        df = pd.read_csv(mapping_path, sep="\t")
        arr = [path.split(os.sep)[-3] for path in df["t1"]]
        df["sub"] = arr
        input_dataset = []
        output_dataset = []
        nb_subjects = len(arr)
        with progressbar.ProgressBar(max_value=nb_subjects,
                                     redirect_stdout=True) as bar:
            for cnt, sub in enumerate(arr):

                logger.debug("Processing {0}...".format(sub))
                subdata = df.loc[df["sub"] == sub]
                data = []
                for mod in MODALITIES:

                    path = subdata[mod].values[0]
                    data.append(_norm(_crop(nib.load(path).get_data())))

                data = np.asarray(data)
                input_dataset.append(data)
                data = []
                path = subdata["mask"].values[0]
                masks = nib.load(path).get_data().astype(int)
                for i in range(4):
                    _arr = masks[:, :, :, i]
                    data.append(_crop(_arr == 1))
                data = np.asarray(data)
                output_dataset.append(data)
                bar.update(cnt)

            input_dataset = np.asarray(input_dataset)
            output_dataset = np.asarray(output_dataset)

            dataset_desc = pd.DataFrame(arr, columns=["participant_id"])

            np.save(input_path, input_dataset)
            np.save(output_path, output_dataset)
            dataset_desc.to_csv(desc_path, sep="\t")

    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)

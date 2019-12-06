# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
# @Author: Zhou Kai
# @GitHub: https://github.com/athon2
# @Date: 2018-11-30 09:53:44
##########################################################################

"""
Module that provides functions to prepare the Brats dataset.
"""

# Imports
from collections import namedtuple
import os
import numpy as np
import pandas as pd
import nibabel as nib
import progressbar


# Global parameters
MODALITIES = ("t1", "t1ce", "t2", "flair")
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])


def fetch_brats(datasetdir):
    """ Fetch/prepare the Brats dataset for pynet.

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
    mapping_path = os.path.join(datasetdir, "name_mapping.csv")
    if not os.path.isfile(mapping_path):
        raise ValueError(
            "You must first download the Brats data in the '{0}' folder "
            "following the 'https://www.med.upenn.edu/sbia/brats2018/"
            "registration.html' instructions.".format(datasetdir))
    desc_path = os.path.join(datasetdir, "pynet_brats.tsv")
    input_path = os.path.join(datasetdir, "pynet_brats_inputs.npy")
    output_path = os.path.join(datasetdir, "pynet_brats_outputs.npy")
    if not os.path.isfile(desc_path):
        df = pd.read_csv(path, sep=",")
        arr = df[["BraTS_2019_subject_ID", "Grade"]].values
        input_dataset = []
        output_dataset = []
        nb_subjects = len(arr)
        with progressbar.ProgressBar(max_value=nb_subjects,
                                     redirect_stdout=True) as bar:
            for cnt, (sid, grade) in enumerate(arr):
                datadir = os.path.join(dirname, grade, sid)
                data = []
                for mod in MODALITIES:
                    path = os.path.join(
                        datadir, "{0}_{1}.nii.gz".format(sid, mod))
                    data.append(nib.load(path).get_data())
                data = np.asarray(data)
                input_dataset.append(data)
                path = os.path.join(datadir, "{0}_seg.nii.gz".format(sid))
                _arr = nib.load(path).get_data()
                data = []
                for value in (0, 1, 2, 4):
                    data.append(_arr == value)
                data = np.asarray(data)
                output_dataset.append(data)
                bar.update(cnt)
            input_dataset = np.asarray(input_dataset)
            np.save(input_path, input_dataset)
            output_dataset = np.asarray(output_dataset)
            np.save(output_path, output_dataset)
            dataset_desc = pd.DataFrame(
                arr, columns=["participant_id", "grade"])
            dataset_desc.to_csv(desc_path, sep="\t")
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)

# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides functions to prepare the Orientation dataset.
"""

from torch.utils.data import Dataset
from nilearn.image import new_img_like, resample_to_img

# Imports
import os
import json
import glob
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import pandas as pd
import progressbar
import urllib
import tarfile
from PIL import Image


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "height", "width", "labels"])
URL = ("https://deepimaging2019.sciencesconf.org/data/pages/"
       "ge_insa_lyon_datasets_dlss_ho4_data_1.tar")


def fetch_orientation(datasetdir, flatten=False):
    """ Fetch/prepare the orientation dataset for pynet.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    flatten: bool, default False
        return a flatten version of the dataset.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    labels = OrderedDict({
            0: "T1-A",
            1: "T1-S",
            2: "T1-C",
            3: "T2-A",
            4: "T2-S",
            5: "T2-C",
            6: "CT-A",
            7: "CT-S",
            8: "CT-C"
    })
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    tarball = os.path.join(datasetdir, "orientation.tar")
    desc_path = os.path.join(datasetdir, "pynet_orientation.tsv")
    input_path = os.path.join(datasetdir, "pynet_orientation_inputs.npy")
    flat_input_path = os.path.join(
        datasetdir, "pynet_orientation_flat_inputs.npy")
    height, width = (64, 64)
    if not os.path.isfile(desc_path):
        if not os.path.isfile(tarball):
            urllib.request.urlretrieve(URL, tarball)
        else:
            print("Tarball already downloaded!")
        downloaddir = tarball.replace(".tar", "")
        if not os.path.isdir(downloaddir):
            tar = tarfile.open(tarball)
            tar.extractall(path=downloaddir)
            tar.close()
        else:
            print("Archive already opened!")
        files = glob.glob(os.path.join(downloaddir, "*.png"))
        nb_files = len(files)
        data = []
        data_flat = []
        metadata = dict((key, []) for key in ("name", "label"))
        rev_labels = dict((val, key) for key, val in labels.items())
        for path in files:
            basename = os.path.basename(path).replace(".png", "")
            im = Image.open(path)
            arr = np.array(im.getdata())
            data_flat.append(arr.copy())
            arr.shape = (height, width)
            data.append(arr)
            metadata["name"].append(basename)
            metadata["label"].append(rev_labels[basename[-4:]])
        data = np.asarray(data)
        np.save(input_path, data)
        np.save(flat_input_path, data_flat)
        df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
    if flatten:
        input_path = flat_input_path
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, height=height, width=width,
                labels=labels)

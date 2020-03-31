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

# Imports
import os
import logging
import json
import glob
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import skimage.io as skio
import pandas as pd
import progressbar
import urllib
import tarfile
from pynet.datasets import Fetchers


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "height", "width"])
URL = ("https://deepimaging2019.sciencesconf.org/data/pages/"
       "ge_insa_lyon_datasets_camus_dataset_2.tar")
logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_echocardiography(datasetdir):
    """ Fetch/prepare the echocardiography dataset for pynet.

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
    logger.info("Loading echocardiography dataset.")

    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor.
        """
        return np.eye(num_classes, dtype="uint8")[y].transpose(2, 0, 1)
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    tarball = os.path.join(datasetdir, "echocardiography.tar")
    desc_path = os.path.join(datasetdir, "pynet_echocardiography.tsv")
    input_path = os.path.join(datasetdir, "pynet_echocardiography_inputs.npy")
    output_path = os.path.join(
        datasetdir, "pynet_echocardiography_outputs.npy")
    height, width = (256, 256)
    if not os.path.isfile(desc_path):
        if not os.path.isfile(tarball):
            urllib.request.urlretrieve(URL, tarball)
        else:
            logger.info("Tarball already downloaded!")
        downloaddir = tarball.replace(".tar", "")
        if not os.path.isdir(downloaddir):
            tar = tarfile.open(tarball)
            tar.extractall(path=downloaddir)
            tar.close()
        else:
            logger.info("Archive already opened!")
        files = glob.glob(os.path.join(downloaddir, "images", "*.png"))
        nb_files = len(files)
        data = []
        masks = []
        metadata = dict((key, []) for key in ("name", "label"))
        for path in files:
            logger.debug("Processing {0}...".format(path))
            basename = os.path.basename(path)
            im = skio.imread(path)
            im = im / np.max(im)
            mask_path = os.path.join(downloaddir, "masks", basename)
            logger.debug("Processing {0}...".format(mask_path))
            mask = skio.imread(mask_path)
            mask = mask.astype(np.single)
            mask = (mask / 255. * 3.).astype(int)
            mask = to_categorical(y=mask, num_classes=4)
            data.append(im)
            masks.append(mask)
            basename = basename.replace(".png", "")
            metadata["name"].append(basename)
            metadata["label"].append(basename[-2:])
        data = np.asarray(data).astype("float32")
        data = np.expand_dims(data, axis=1)
        masks = np.asarray(masks).astype("float32")
        np.save(input_path, data)
        np.save(output_path, masks)
        df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path, height=height, width=width)

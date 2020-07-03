# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the TCGA-LGG-tif dataset.
"""

# Imports
from collections import OrderedDict
import os
import logging
from collections import namedtuple
import numpy as np
import skimage.io as skio
import pandas as pd
from pynet.datasets import Fetchers
import progressbar
import csv
import glob

# Global parameters
Item = namedtuple("Item", ["input_path", "output_path",
                           "metadata_path", "height", "width"])
URL = "https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation/download"
logger = logging.getLogger("pynet")
height, width = (256, 256)


def read_metadata(metadata_file):
    with open(metadata_file) as f:
        metadata_dict = csv.DictReader(f)
        metadata_dict = {
            row["Patient"].split("_")[-1]: row for row in metadata_dict}
    return metadata_dict


def get_slice_id(fp):
    return int(
        fp.replace("_mask.tif", "").replace(".tif", "").split("_")[-1])


def get_subjects_files(datadir):
    sdata = {}
    for fp in glob.glob(
            os.path.join(datadir, "*", "*", "*_mask.tif")):
        dirname = fp.split(os.sep)[-2]
        _, center, subject, serie = dirname.split("_")
        if subject not in sdata:
            sdata[subject] = {"center": center,
                              "serie": serie,
                              "masks": [],
                              "images": []}
        sdata[subject]["masks"].append(fp)
        sdata[subject]["images"].append(fp.replace("_mask", ""))
    return sdata


@Fetchers.register
def fetch_tcga_lgg_tif(datasetdir):
    """ Fetch/prepare the TCA-LGG-tif dataset for pynet.

    The patient average age was 47 with an almost even split between women and
    men (56 vs. 53, 1 unknown) in our dataset. Histologically, the tumors were
    divided between oligodendroglioma (47), astrocytoma (33), and
    oligoastrocytoma (29). Histology of one tumor was unknown. The data
    included grade II (51) and grade III (58) tumors with grade of one tumor
    unknown.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path',
        'metadata_path', 'height' and 'width'.
    """

    logger.info("Loading TCA-LGG-tif dataset.")
    if not os.path.isdir(datasetdir):
        raise ValueError(
            "You must first download the kaggle dataset at {} and unzip it "
            "to {}.".format(URL, datasetdir))

    metadata_path = os.path.join(datasetdir, "kaggle_3m", "data.csv")
    desc_path = os.path.join(datasetdir, "pynet_tgca-lgg-tif.tsv")
    input_path = os.path.join(datasetdir, "pynet_tgca-lgg-tif_inputs.npy")
    output_path = os.path.join(datasetdir, "pynet_tgca-lgg-tif_outputs.npy")

    if not os.path.isfile(desc_path):
        # parse datasetdir
        sdata = get_subjects_files(datasetdir)
        # parse genetics csv file
        smetadata = read_metadata(metadata_path)
        input_dataset = []
        output_dataset = []
        metadata = OrderedDict((key, []) for key in (
            "participant_id", "slice_id", "center", "serie"))
        with progressbar.ProgressBar(max_value=len(sdata),
                                     redirect_stdout=True) as bar:
            for cnt, (subject, subject_data) in enumerate(sdata.items()):
                logger.debug("Processing {0}...".format(subject))

                for impath in subject_data["images"]:
                    # (height, width, (precontrast, flair, postcontrast))
                    # -> ((precontrast, flair, postcontrast), height, width)
                    im = skio.imread(impath).transpose(2, 0, 1)
                    input_dataset.append(im)

                    # Get subject genetics metadata
                    metadata["participant_id"].append(subject)
                    metadata["slice_id"].append(
                        get_slice_id(impath))
                    metadata["center"].append(subject_data["center"])
                    metadata["serie"].append(subject_data["serie"])
                    for meta_name, meta_value in smetadata[subject].items():
                        metadata.setdefault(meta_name, []).append(meta_value)

                for impath in subject_data["masks"]:
                    im = skio.imread(impath)[np.newaxis, ...]
                    im[im == 255] = 1
                    assert set(im.ravel().tolist()).issubset({
                        0, 1})
                    output_dataset.append(im)

                bar.update(cnt)

            input_dataset = np.asarray(input_dataset)
            np.save(input_path, input_dataset)
            output_dataset = np.asarray(output_dataset)
            np.save(output_path, output_dataset)

            df = pd.DataFrame.from_dict(metadata)
            df.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path, height=height, width=width)

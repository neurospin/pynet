# -*- coding: utf-8 -*-
########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
########################################################################

"""
Module provides functions to prepare different datasets from HCP.
  1) the T1 and associated brain masks.
"""

# Imports
import os
import logging
from collections import namedtuple
import boto3
from botocore.exceptions import NoCredentialsError
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from pynet.datasets import Fetchers


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])
logger = logging.getLogger("pynet")


@Fetchers.register
def fetch_hcp_brain(datasetdir, low=False, small=True):
    """ Fetch/prepare the HCP T1/brain mask dataset for pynet.

    Go to 'https://db.humanconnectome.org' and get an account and log in.
    Then, click on the Amazon S3 button that should give you a key pair.
    Then use 'aws configure' to add this to our machine.
    AWS Access Key ID: ****************
    AWS Secret Access Key: ****************
    Default region name: eu-west-3
    Default output format: json

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    low: bool, default False
        set images in low resolution.
    small: bool, default True
        fetch 45 brains if true, else 1200 brains.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    logger.info("Loading HCP brain dataset...")
    desc_path = os.path.join(datasetdir, "pynet_hcp_brain.tsv")
    input_path = os.path.join(datasetdir, "pynet_hcp_brain_inputs.npy")
    output_path = os.path.join(datasetdir, "pynet_hcp_brain_outputs.npy")

    if not os.path.isfile(desc_path):
        client = boto3.client("s3")
        paginator = client.get_paginator("list_objects")
        prefix = "HCP_1200/"
        result = paginator.paginate(
            Bucket="hcp-openaccess", Delimiter="/", Prefix=prefix)
        try:
            subjects_prefix = list(result.search("CommonPrefixes"))
        except NoCredentialsError:
            msg = """
            Go to 'https://db.humanconnectome.org' and get an account and
            log in.
            Then, click on the Amazon S3 button that should give you a key
            pair.
            Then use 'aws configure' to add this to our machine.
            AWS Access Key ID: ****************
            AWS Secret Access Key: ****************
            Default region name: eu-west-3
            Default output format: json
            """
            raise ValueError(msg)
        if small:
            subjects_prefix = subjects_prefix[:45]

        images = []
        masks = []
        metadata = dict((key, []) for key in ("name", "modality"))
        for subject in subjects_prefix:
            subject_prefix = subject["Prefix"]
            logger.info("  subject: {0}".format(subject_prefix))
            data = get_hcp_data(datasetdir, subject_prefix, "T1w", low)
            metadata["name"].append(subject_prefix[11: -1])
            metadata["modality"].append("T1w")
            images.append(data["image"])
            masks.append(data["mask"].astype(int))
            data = get_hcp_data(datasetdir, subject_prefix, "MNINonLinear",
                                low)
            metadata["name"].append(subject_prefix[11: -1])
            metadata["modality"].append("MNINonLinear")
            images.append(data["image"])
            masks.append(data["mask"].astype(int))
        images = np.asarray(images)
        masks = np.asarray(masks)
        images = np.expand_dims(images, axis=1)
        masks = np.expand_dims(masks, axis=1)
        np.save(input_path, images)
        np.save(output_path, masks)
        df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
        logger.info("Done.")
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)


def load_image(filename, low=False):
    """ Load an MRI image.

    High resolution images are resampled to (256, 312, 256) and low resolution
    images are resampled to (32, 40, 32) which can be divided by 8.

    Parameters
    ----------
    filename: str
        file to be loaded.
    low: bool, default False
        set image in low resolution.

    Returns
    -------
    img_data: np.array
        loaded image.
    """
    img = nib.load(filename)
    img_data = img.get_data()
    img_data = np.append(
        img_data[2:-2, :, 2:-2], np.zeros((256, 1, 256)), axis=1)
    if low:
        img_data = ndimage.zoom(img_data, 1. / 8., order=0)
        img_data = np.append(img_data, np.zeros((32, 1, 32)), axis=1)
    return img_data


def get_hcp_data(datasetdir, subject_prefix, modality, low):
    """ Get the requested data.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    subject_prefix: str
        subject path.
    modality: str
        type of image to be extracted ('T1w' or 'MNINonLinear').
    low: bool
        set image in low resolution.

    Returns
    -------
    data: dict
        the loaded data.
    """
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("hcp-openaccess")
    mapping = {
        "T1w": {
            "image": "T1w_acpc_dc_restore.nii.gz",
            "mask": "brainmask_fs.nii.gz"},
        "MNINonLinear": {
            "image": "T1w_restore.nii.gz",
            "mask": "brainmask_fs.nii.gz"}
    }
    if modality not in mapping:
        raise ValueError("Unexpected modality '{0}'. Valid modalities "
                         "are: {1}".format(modality, mapping.keys()))
    data = {}
    for key, basename in mapping[modality].items():
        url = subject_prefix + "/".join([modality, basename])
        pattern = url.split("/")
        destfile = os.path.join(datasetdir, *pattern)
        if not os.path.isfile(destfile):
            logger.info("  url: {0}".format(url))
            logger.info("  dest: {0}".format(destfile))
            dirname = os.path.dirname(destfile)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            obj = s3.Object(bucket, url)
            bucket.download_file(obj.key, destfile)
        data[key] = load_image(destfile, low)
    return data

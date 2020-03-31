# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare registration dataset.
"""

# Imports
import os
import json
import h5py
import glob
import urllib
import shutil
import requests
import logging
import numpy as np
import scipy
import zipfile
import skimage
from collections import namedtuple
import pandas as pd
from pynet.datasets import Fetchers


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])
URL = "https://docs.google.com/uc?export=download"
ID = "1rJtP9M1N3lSjNzJ5kIzRrrwPe1bWCfXB"
ATLAS = ("https://github.com/voxelmorph/voxelmorph/raw/master/data/"
         "atlas_norm.npz")
logger = logging.getLogger("pynet")


def download_file_from_google_drive(destination):
    session = requests.Session()
    response = session.get(URL, params={"id": ID}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"id": ID, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def wl_normalization(img, w=290, l=120):
    img = skimage.exposure.rescale_intensity(
        img, in_range=(l - w / 2, l + w / 2), out_range=(0, 255))
    return img.astype(np.uint8)


def crop(arr, bound_l, bound_r, target_shape, order=1):
    cropped = arr[bound_l[0]: bound_r[0], bound_l[1]: bound_r[1],
                  bound_l[2]: bound_r[2]]
    return scipy.ndimage.zoom(
        cropped, np.array(target_shape) / np.array(cropped.shape),
        order=order)


def crop_mask(volume, segmentation, target_shape=(128, 128, 128)):
    indices = np.array(np.nonzero(segmentation))
    bound_r = np.max(indices, axis=-1)
    bound_l = np.min(indices, axis=-1)
    box_size = bound_r - bound_l + 1
    padding = np.maximum((box_size * 0.1).astype(np.int32), 5)
    bound_l = np.maximum(bound_l - padding, 0)
    bound_r = np.minimum(bound_r + padding + 1, segmentation.shape)
    return wl_normalization(crop(volume, bound_l, bound_r, target_shape))


@Fetchers.register
def fetch_registration(datasetdir):
    """ Fetch/prepare the registration dataset for pynet.

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
    logger.info("Loading registration dataset...")
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_registration.tsv")
    input_path = os.path.join(datasetdir, "pynet_registration_inputs.npy")
    if not os.path.isfile(desc_path):
        logger.debug("Processing {0}...".format(URL))
        zfile = os.path.join(datasetdir, "brain_train.zip")
        if not os.path.isfile(zfile):
            download_file_from_google_drive(zfile)
        else:
            logger.info("ZIP already downloaded!")
        afile = os.path.join(datasetdir, "atlas_norm.npz")
        if not os.path.isfile(afile):
            response = requests.get(ATLAS, stream=True)
            with open(afile, "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
        else:
            logger.info("ATLAS already downloaded!")
        downloadir = os.path.join(datasetdir, "datasets")
        if not os.path.isdir(downloadir):
            with zipfile.ZipFile(zfile, "r") as zip_ref:
                zip_ref.extractall(downloadir)
        else:
            logger.info("Archive already opened!")
        # TODO fix that
        atlas = np.load(afile)["vol"]
        atlas *= 7000
        logger.debug("Atlas {0}...".format(atlas.shape))
        mask = atlas.astype(int)
        mask[mask > 0] = 1
        # atlas_norm = crop_mask(atlas, mask, target_shape=(128, 128, 128))
        # logger.debug("Norm atlas {0}...".format(atlas_norm.shape))
        try:
            import nibabel
            im = nibabel.Nifti1Image(atlas_norm, np.eye(4))
            nibabel.save(im, os.path.join(datasetdir, "atlas_norm.nii.gz"))
        except:
            pass
        files = glob.glob(os.path.join(downloadir, "*.h5"))
        all_arrs = []
        metadata = dict((key, []) for key in ("subjects", "centers", "studies",
                                              "keys"))
        for h5file in files:
            logger.debug("Processing {0}...".format(h5file))
            open_file = h5py.File(h5file, "r")
            study = os.path.basename(h5file).replace(".h5", "")
            for key in open_file.keys():
                if key in metadata["keys"]:
                    raise ValueError(
                        "Key '{0}' appears multiple time.".format(key))
                try:
                    center, sid = key.split("-")
                except:
                    center = "na"
                    sid = key
                logger.debug("Processing key {0} ({1}-{2})...".format(
                    key, sid, center))
                data = open_file[key]["volume"]
                metadata["subjects"].append(sid)
                metadata["centers"].append(center)
                metadata["studies"].append(study)
                metadata["keys"].append(key)
                all_arrs.append(np.array(data))
        data = np.asarray(all_arrs)
        try:
            import nibabel
            im = nibabel.Nifti1Image(
                np.transpose(data, (1, 2, 3, 0)), np.eye(4))
            nibabel.save(im, os.path.join(datasetdir, "data.nii.gz"))
        except:
            pass
        atlas_norm = data[0]
        atlas_norm = np.expand_dims(atlas_norm, axis=0)
        atlas_norm = np.repeat(atlas_norm, len(data), axis=0)
        data = np.expand_dims(data, axis=1)
        atlas_norm = np.expand_dims(atlas_norm, axis=1)
        logger.debug("Data: {0}".format(data.shape))
        logger.debug("Atlas: {0}".format(atlas_norm.shape))
        data = np.concatenate((data, atlas_norm), axis=1)
        logger.debug("Input: {0}-{1}".format(data.shape, data.dtype))
        np.save(input_path, data)
        df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=None)

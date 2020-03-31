import os
from collections import namedtuple

import boto3
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])


def load_image(filename, low=False):
    """

    Parameters
    ----------
    filename: str
        file to be loaded
    low: bool
        set image in low resolution

    Returns
    -------
    img_data: np.array
        image to be used
    """
    img = nib.load(filename)
    img_data = img.get_data()
    # set img_data.shape=(256,312,256) which can be divided by 8
    img_data = np.append(img_data[2:-2, :, 2:-2], np.zeros((256, 1, 256)), axis=1)

    if low:
        img_data = ndimage.zoom(img_data, 1 / 8, order=0)
        # set img_data.shape=(32,40,32) which can also be divided by 8
        img_data = np.append(img_data, np.zeros((32, 1, 32)), axis=1)
    return img_data


def extract_hcp_data(datasetdir, data, masks, subject, modality, low, metadata):
    """

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    data: list
        current list of brains
    masks: list
        current list of masks
    subject: str
        subject id
    modality: str
        type of image to be extracted ('T1w' or 'MNINonLinear')
    low: bool
        set image in low resolution
    metadata: dictionary
        subject id and image modality

    Returns
    -------
    data: list
        updated list of brains
    masks: list
        updated list of masks
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')
    if modality == 'T1w':
        obj = s3.Object(bucket, os.path.join(subject, modality, 'T1w_acpc_dc.nii.gz'))
        if not os.path.isfile(os.path.join(datasetdir, subject, 'brainmask_acpc.nii.gz')):
            bucket.download_file(obj.key, os.path.join(datasetdir, subject, 'brainmask_acpc.nii.gz'))

        im = load_image(os.path.join(datasetdir, subject, 'T1w_acpc_dc.nii.gz'), low)
        data.append(im)

        obj = s3.Object(bucket, os.path.join(subject, modality, 'brainmask_fs.nii.gz'))
        if not os.path.isfile(os.path.join(datasetdir, subject, 'brainmask_acpc.nii.gz')):
            bucket.download_file(obj.key, os.path.join(datasetdir, subject, 'brainmask_acpc.nii.gz'))

        mask = load_image(os.path.join(datasetdir, subject, 'brainmask_acpc.nii.gz'), low)
        masks.append(mask)


    elif modality == 'MNINonLinear':
        obj = s3.Object(bucket, os.path.join(subject, modality, 'T1w.nii.gz'))
        if not os.path.isfile(os.path.join(datasetdir, subject, 'T1w.nii.gz')):
            bucket.download_file(obj.key, os.path.join(datasetdir, subject, 'T1w.nii.gz'))

        im = load_image(os.path.join(datasetdir, subject, 'T1w.nii.gz'), low)
        data.append(im)

        obj = s3.Object(bucket, os.path.join(subject, modality, 'brainmask_fs.nii.gz'))
        if not os.path.isfile(os.path.join(datasetdir, subject, 'brainmask_fs.nii.gz')):
            bucket.download_file(obj.key, os.path.join(datasetdir, subject, 'brainmask_fs.nii.gz'))

        mask = load_image(os.path.join(datasetdir, subject, 'brainmask_acpc.nii.gz'), low)
        masks.append(mask)
    else:
        raise ValueError("'{0}' is not a valid modality. 'T1w' or 'MNINonLinear' expected".format(
            modality))

    metadata["name"].append(subject[11:-1])
    metadata["modality"].append(modality)

    return data, mask, metadata


def fetch_data(datasetdir, low=False, small=False):
    """ Fetch/prepare the dataset for pynet.
    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    low: bool
        set images in low resolution
    small: bool
        fetch 45 brains if true, else 1200 brains
    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """

    desc_path = os.path.join(datasetdir, "human_brain.tsv")
    input_path = os.path.join(datasetdir, "human_brain_inputs.npy")
    output_path = os.path.join(datasetdir, "human_brain_outputs.npy")

    paginator = boto3.client('s3').get_paginator('list_objects')
    if small:
        prefix = 'HCP_Retest/'
    else:
        prefix = 'HCP_1200/'
    result = paginator.paginate(Bucket='hcp-openaccess', Delimiter='/', Prefix=prefix)
    subjects_prefix = result.search('CommonPrefixes')

    data = []  # type: list[np.ndarray]
    masks = []  # type: list[np.ndarray]
    metadata = dict((key, []) for key in ("name", "modality"))

    if not os.path.isdir(os.path.join(datasetdir, prefix)):
        os.mkdir(os.path.join(datasetdir, prefix))
    for subject in subjects_prefix:
        if not os.path.isdir(os.path.join(datasetdir, subject['Prefix'])):
            os.mkdir(os.path.join(datasetdir, subject['Prefix']))

        data, mask, metadata = extract_hcp_data(datasetdir, data, masks, subject['Prefix'], 'T1w', low, metadata)
        data, mask, metadata = extract_hcp_data(datasetdir, data, masks, subject['Prefix'], 'MNINonLinear', low,
                                                metadata)

    data = np.expand_dims(data, axis=1)
    masks = np.expand_dims(masks, axis=1)
    np.save(input_path, data)
    np.save(output_path, masks)
    df = pd.DataFrame.from_dict(metadata)
    df.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)

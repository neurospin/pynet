import os
from collections import namedtuple

import boto3
import nibabel as nib
import numpy as np
import pandas as pd

Item = namedtuple("Item", ["input_path", "output_path", "metadata_path"])


def nii_to_np_arr(filename, low=False, binary=False):
    img = nib.load(filename)
    img_data = img.get_data()
    img_data = np.append(img_data[2:-2, :, 2:-2], np.zeros((256, 9, 256)), axis=1)
    if low:
        img_data = img_data.reshape(32, 8, 40, 8, 32, 8).mean(axis=(1, 3, 5))

    if binary:
        img_data = 1 * (img_data > img_data.mean())
    return img_data


def extract_hcp_data(datasetdir, data, masks, s3, bucket, subject, label, low, metadata):
    if label == 'T1w':
        obj = s3.Object(bucket, subject + label + '/T1w_acpc_dc.nii.gz')
    else:
        obj = s3.Object(bucket, subject + label + '/T1w.nii.gz')

    if not os.path.isdir(datasetdir + subject + label):
        os.mkdir(datasetdir + subject + label)

    if not os.path.isfile(datasetdir + obj.key):
        bucket.download_file(obj.key, datasetdir + obj.key)
    im = nii_to_np_arr(datasetdir + obj.key, low)
    data.append(im)

    obj = s3.Object(bucket, subject + label + '/brainmask_fs.nii.gz')
    if not os.path.isfile(datasetdir + obj.key):
        bucket.download_file(obj.key, datasetdir + obj.key)

    mask = nii_to_np_arr(datasetdir + obj.key, low, binary=True)
    masks.append(mask)

    metadata["name"].append(subject[11:-1])
    metadata["label"].append(label)

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

    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')
    paginator = boto3.client('s3').get_paginator('list_objects')
    if small:
        prefix = 'HCP_Retest/'
    else:
        prefix = 'HCP_1200/'
    result = paginator.paginate(Bucket='hcp-openaccess', Delimiter='/', Prefix=prefix)
    subjects_prefix = result.search('CommonPrefixes')

    data = []
    masks = []
    metadata = dict((key, []) for key in ("name", "label"))

    if not os.path.isdir(datasetdir + prefix):
        os.mkdir(datasetdir + prefix)
    for subject in subjects_prefix:
        if not os.path.isdir(datasetdir + subject['Prefix']):
            os.mkdir(datasetdir + subject['Prefix'])
        data, mask, metadata = extract_hcp_data(datasetdir, data, masks, s3, bucket, subject['Prefix'], 'T1w', low,
                                                metadata)
        data, mask, metadata = extract_hcp_data(datasetdir, data, masks, s3, bucket, subject['Prefix'], 'MNINonLinear',
                                                low,
                                                metadata)

    data = np.expand_dims(data, axis=1)
    masks = np.expand_dims(masks, axis=1)
    np.save(input_path, data)
    np.save(output_path, masks)
    df = pd.DataFrame.from_dict(metadata)
    df.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=output_path,
                metadata_path=desc_path)

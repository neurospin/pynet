# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the GradCam dataset.
"""

# Imports
import os
import json
import urllib
import shutil
import requests
import numpy as np
from torchvision import transforms
from torchvision import datasets
from collections import namedtuple
import pandas as pd


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])
URLS = [
    "https://miro.medium.com/max/419/1*kc-k_j53HOJH_sifhg4lHg.jpeg",
    "https://miro.medium.com/max/500/1*506ySAThs6pItevFqZF-4g.jpeg",
    "https://miro.medium.com/max/500/1*XbnzdczNru6HsX6qPZaXLg.jpeg",
    "https://miro.medium.com/max/384/1*oRpjlGC3sUy5yQJtpwclwg.jpeg",
    "https://miro.medium.com/max/500/1*EQ3JBr2vGPuovYFyh6mQeQ.jpeg"
]


def fetch_gradcam(datasetdir, inception=False):
    """ Fetch/prepare the GradCam dataset for pynet.

    Parameters
    ----------
    datasetdir: str
        the dataset destination folder.
    inception: bool, default True
        if set apply the inception ttransforms on the inputs.

    Returns
    -------
    item: namedtuple
        a named tuple containing 'input_path', 'output_path', and
        'metadata_path'.
    """
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    labels_url = (
        "https://s3.amazonaws.com/deep-learning-models/image-models/"
        "imagenet_class_index.json")
    with urllib.request.urlopen(labels_url) as response:
        labels = dict(
            (key, val)
            for key, val in json.loads(response.read().decode()).items())
    desc_path = os.path.join(datasetdir, "pynet_gradcam.tsv")
    input_path = os.path.join(datasetdir, "pynet_gradcam_inputs.npy")
    incep_input_path = os.path.join(
        datasetdir, "pynet_gradcam_incep_inputs.npy")
    if not os.path.isfile(desc_path):
        imagedir = os.path.join(datasetdir, "images")
        if not os.path.isdir(imagedir):
            os.mkdir(imagedir)
        metadata = dict((key, []) for key in ("name", ))
        for cnt, url in enumerate(URLS):
            ext = url.split(".")[-1]
            name = "image{0}".format(cnt)
            imagefile = os.path.join(imagedir, name + "." + ext)
            metadata["name"].append(name)
            if not os.path.isfile(imagefile):
                response = requests.get(url, stream=True)
                with open(imagefile, "wb") as out_file:
                    shutil.copyfileobj(response.raw, out_file)
                del response
            else:
                print("Image '{0}' already downloaded.".format(imagefile))
        transform = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(root=datasetdir, transform=transform)
        data = []
        for item in dataset:
            data.append(item[0].numpy())
        data = np.asarray(data)
        np.save(input_path, data)
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        data = []
        for item in dataset:
            data.append(item[0].numpy())
        data = np.asarray(data)
        np.save(incep_input_path, data)
        df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
    if inception:
        input_path = incep_input_path
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=labels)

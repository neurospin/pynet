# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the Brats dataset.
"""

# Imports
import os
import json
import torchvision
import torchvision.transforms as transforms
from collections import namedtuple
from collections import OrderedDict
import numpy as np
import pandas as pd
import urllib


# Global parameters
Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])


def fetch_cifar(datasetdir):
    """ Fetch/prepare the CIFAR dataset for pynet.

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
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
               "ship", "truck")
    labels = OrderedDict((key, val) for key, val in enumerate(classes))
    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
    desc_path = os.path.join(datasetdir, "pynet_cifar.tsv")
    input_path = os.path.join(datasetdir, "pynet_cifar_inputs.npy")
    if not os.path.isfile(desc_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=datasetdir,
            train=True,
            download=True,
            transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root=datasetdir,
            train=False,
            download=True,
            transform=transform)
        metadata = dict((key, []) for key in ("label", "category"))
        data = []
        for loader in (trainset, testset):
            for arr, label in loader:
                data.append(arr.numpy())
                metadata["label"].append(label)
                metadata["category"].append(labels[label][1])
        data = np.asarray(data)
        np.save(input_path, data)
        df = pd.DataFrame.from_dict(metadata)
        df.to_csv(desc_path, sep="\t", index=False)
    return Item(input_path=input_path, output_path=None,
                metadata_path=desc_path, labels=labels)

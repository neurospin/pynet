# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides functions to prepare the DSprites dataset.

beta-vae: Learning basic visual concepts with a constrained variational
framework, Higgins, International Conference on Learning Representations, 2017.

Code: https://github.com/YannDubs/disentangling-vae
"""

# Imports
import os
import logging
import subprocess
import numpy as np
from pynet.datasets.core import DataItem
from torch.utils.data import Dataset


# Global parameters
logger = logging.getLogger("pynet")


class DSprites(Dataset):
    """ Disentanglement test Sprites dataset.

    Procedurally generated 2D shapes, from 6 disentangled latent factors.
    This dataset uses 6 latents, controlling the color, shape, scale,
    rotation and position of a sprite.
    All possible variations of the latents are present. Ordering along
    dimension 1 is fixed and can be mapped back to the exact latent values
    that generated that image. Pixel outputs are different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2
    """
    urls = {
        "train":
            "https://github.com/deepmind/dsprites-dataset/blob/master/"
            "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ("shape", "scale", "orientation", "posX", "posY")
    img_size = (64, 64)

    def __init__(self, datasetdir, size=None, **kwargs):
        """ Init class.

        Latent values of length 6, that gives the value of each factor of
        variation.

        Parameters
        ----------
        datasetdir: string
            the dataset destination folder.
        size: int, default None
            the size of the dataset, default use all images available.

        Returns
        -------
        item: namedtuple
            a named tuple containing 'input_path', and 'metadata_path'.
        """
        super(DSprites, self).__init__(**kwargs)
        self.datasetdir = datasetdir
        self.dsprites_file = os.path.join(
            self.datasetdir, DSprites.files["train"])
        self.download()
        dataset = np.load(self.dsprites_file)
        if size is None:
            size = len(dataset["imgs"])
        size = min(size, len(dataset["imgs"]))
        index = np.arange(size)
        np.random.shuffle(index)
        self.imgs = dataset["imgs"][index]
        self.lat_values = dataset["latents_values"][index]
        self.n_samples = len(self.imgs)

    def download(self):
        """ Download the dataset.
        """
        if not os.path.isdir(self.datasetdir):
            os.makedirs(self.datasetdir)
        if not os.path.isfile(self.dsprites_file):
            subprocess.check_call(["curl", "-L", DSprites.urls["train"],
                                   "--output", self.dsprites_file])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """ Get the image at position 'idx'.

        Returns
        -------
        out: DataItem
            input/output tensor in [0, 1] of shape 'img_size'.
        """
        data = np.expand_dims(self.imgs[idx], axis=0)
        return DataItem(inputs=data, outputs=data, labels=None)

# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that privides common spatial and intensity data augmentation tools.
"""

# Import
from collections import namedtuple
import numpy as np
from .spatial import affine
from .spatial import flip
from .spatial import deformation
from .spatial import padd
from .spatial import downsample
from .intensity import add_blur
from .intensity import add_noise
from .intensity import add_ghosting
from .intensity import add_spike
from .intensity import add_biasfield
from .intensity import add_motion


class Transformer(object):
    """ Class that can be used to register an sequence of transsformations
    """
    Transform = namedtuple("Transform", ["transform", "params", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1, **kwargs):
        """ Register a new transformation.

        Parameters
        ----------
        transform: callable
            the transformation function.
        probability: float, default 1
            the transform is applied with the specified probability.
        kwargs
            the transformation function parameters.
        """
        trf = self.Transform(
            transform=transform, params=kwargs, probability=probability)
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.

        Parameters
        ----------
        arr: array
            the input data.

        Returns
        -------
        transformed: array
            the transformed input data.
        """
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                arr = trf.transform(arr, **trf.params)
        return arr

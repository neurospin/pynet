# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that defines common transformations that can be applied when the dataset
is loaded.
"""


# Third party import
import numpy as np


class ZeroPadding(object):
    """ A class to zero pad an image.
    """
    def __init__(self, shape):
        """ Initialize the instance.

        Parameters
        ----------
        shape: list of int
            the desired shape.
        """
        self.shape = shape
        

    def __call__(self, arr, fill_value=0):
        """ Zero fill an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array.
        fill_value: int
            the value used to fill the array.

        Returns
        -------
        fill_arr: np.array
            the zero padded array.
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append((0, 0))
        fill_arr = np.pad(arr, padding, mode="constant",
                          constant_values=fill_value)
        return fill_arr


class Downsample(object):
    """ A class to downsample an array.
    """
    def __init__(self, scale):
        """ Initialize the instance.

        Parameters
        ----------
        scale: int
            the downsampling scale factor in all directions.
        """
        self.scale = scale

    def __call__(self, arr):
        """ Downsample an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array
        scale: int
            the downsampling scale factor in all directions.

        Returns
        -------
        down_arr: np.array
            the downsampled array.
        """
        slices = []
        for cnt, orig_i in enumerate(arr.shape):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale))
        down_arr = arr[tuple(slices)]

        return down_arr

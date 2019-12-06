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

# Imports
import collections
import numpy as np


class Padding(object):
    """ A class to pad an image.
    """
    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.

        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = shape
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels

    def __call__(self, arr):
        """ Fill an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array.

        Returns
        -------
        fill_arr: np.array
            the zero padded array.
        """
        if len(arr.shape) - len(self.shape) == 1:
            data = []
            for _arr, _fill_value in zip(arr, self.fill_value):
                data.append(self._apply_padding(_arr, _fill_value))
            return np.asarray(data)
        elif len(arr.shape) - len(self.shape) == 0:
            return self._apply_padding(arr, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr, fill_value):
        """ See Padding.__call__().
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
    def __init__(self, scale, with_channels=True):
        """ Initialize the instance.

        Parameters
        ----------
        scale: int
            the downsampling scale factor in all directions.
        with_channels: bool, default True
            if set expect the array to contain the channels in first dimension.
        """
        self.scale = scale
        self.with_channels = with_channels

    def __call__(self, arr):
        """ Downsample an array to fit the desired shape.

        Parameters
        ----------
        arr: np.array
            an input array

        Returns
        -------
        down_arr: np.array
            the downsampled array.
        """
        if self.with_channels:
            data = []
            for _arr in arr:
                data.append(self._apply_downsample(_arr))
            return np.asarray(data)
        else:
            return self._apply_downsample(arr)

    def _apply_downsample(self, arr):
        """ See Downsample.__call__().
        """
        slices = []
        for cnt, orig_i in enumerate(arr.shape):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale))
        down_arr = arr[tuple(slices)]

        return down_arr

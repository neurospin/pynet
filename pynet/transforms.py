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
from torchvision import transforms


class RandomFlipDimensions(object):
    """ Apply a random mirror flip for all axes with a defined probability.
    """
    def __init__(self, ndims, proba, with_channels=True):
        """ Initilaize the class.

        Parameters
        ----------
        ndims: int
            the number of dimensions.
        proba: float
            apply flip on each axis with this probability [0 - 1].
        with_channels: bool, default True
            if set expect the array to contain the channels in first dimension.
        """
        if proba < 0 or proba > 1:
            raise ValueError("The probabilty must be in [0 - 1].")
        self.ndims = ndims
        self.proba = proba
        self.with_channels = with_channels

    def _random_flip(self):
        """ Generate a random axes flip.
        """
        axis = []
        for dim in range(self.ndims):
            if np.random.choice([True, False], p=[self.proba, 1 - self.proba]):
                axis.append(dim)
        return tuple(axis)

    def __call__(self, arr):
        """ Flip an array axes randomly.

        Parameters
        ----------
        arr: np.array
            an input array.

        Returns
        -------
        flip_arr: np.array
            the fliped array.
        """
        if self.with_channels:
            data = []
            flip = self._random_flip()
            for _arr in arr:
                data.append(np.flip(_arr, axis=flip))
            return np.asarray(data)
        else:
            return np.flip(arr, axis=self._random_flip())


class Offset(object):
    """ Apply an intensity offset (shift and scale) on input channels.
    """
    def __init__(self, nb_channels, factor):
        """ Initilaize the class.

        Parameters
        ----------
        nb_channels: int
            the number of channels.
        factor: float
            the offset scale factor [0 - 1].
        """
        if factor < 0 or factor > 1:
            raise ValueError("The offset factor must be in [0 - 1].")
        self.nb_channels = nb_channels
        self.factor = factor

    def _random_offset(self):
        """ Generate a random offset factor.
        """
        return (2 * self.factor * np.random.random(self.nb_channels) +
                (1 - self.factor))

    def __call__(self, arr):
        """ Normalize an array.

        Parameters
        ----------
        arr: np.array
            an input array.

        Returns
        -------
        offset_arr: np.array
            the rescaled array.
        """
        assert len(arr) == self.nb_channels
        mean_scale_factors = self._random_offset()
        std_scale_factors = self._random_offset()
        data = []
        for _arr, _mfactor, _sfactor in zip(
                arr, mean_scale_factors, std_scale_factors):
            logical_mask = (_arr != 0)
            mean = _arr[logical_mask].mean()
            std = _arr[logical_mask].std()
            data.append((_arr - (mean * _mfactor)) / (std * _sfactor))
        return np.asarray(data)


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

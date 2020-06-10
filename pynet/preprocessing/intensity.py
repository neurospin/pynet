# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to normalize intensities.
Code: https://github.com/jcreinhold/intensity-normalization
"""

# Import
import warnings
import numpy as np
from .hist import get_largest_mode
from .hist import get_last_mode
from .hist import get_first_mode


def rescale(arr, mask=None, percentiles=(0, 100), dynamic=(0, 1)):
    """ Performs a rescale of the image intensities to a certain range.

    Parameters
    ----------
    arr: array
        the input data.
    mask: array, default None
        the brain mask.
    percentiles: 2-uplet, default (0, 100)
        percentile values of the input image that will be mapped. This
        parameter can be used for contrast stretching.
    dynamic: 2-uplet, default (0, 1)
        the intensities range of the rescaled data.

    Returns
    -------
    rescaled: array
        the rescaled input data.
    """
    if mask is not None:
        values = arr[mask]
    else:
        values = arr
    cutoff = np.percentile(values, percentiles)
    rescaled = np.clip(arr, *cutoff)
    rescaled -= rescaled.min()  # [0, max]
    array_max = rescaled.max()
    if array_max == 0:
        warnings.warn("Rescaling not possible due to division by zero.")
        return arr
    rescaled /= rescaled.max()  # [0, 1]
    out_range = dynamic[1] - dynamic[0]
    rescaled *= out_range  # [0, out_range]
    rescaled += dynamic[0]  # [out_min, out_max]
    return rescaled


def zscore_normalize(arr, mask=None):
    """ Performs a batch Z-score normalization.

    Parameters
    ----------
    arr: array
        the input data.
    mask: array, default None
        the brain mask.

    Returns
    -------
    normalized: array
        the normalized input data.
    """
    if mask is not None:
        values = arr[mask == 1]
    else:
        values = arr
    mean = values.mean()
    std = values.std()
    return (arr - mean) / std


def kde_normalize(arr, mask=None, modality="T1w", norm_value=1):
    """ Use kernel density estimation to find the peak of the white
    matter in the histogram of a skull-stripped image. Then normalize
    intensitites to a normalization value.

    Parameters
    ----------
    arr: array
        the input data.
    mask: array, default None
        the brain mask.
    modality str, default 'T1w'
        the modality (T1w, T2w, FLAIR, MD, last, largest, first).
    norm_value: float, default 1
        the new intensity value for the detected WM peak.

    Returns
    -------
    normalized: array
        the normalized input data.
    """
    if mask is not None:
        values = arr[mask == 1]
    else:
        values = arr[arr > arr.mean()]
    if modality.lower() in ["t1w", "flair", "last"]:
        wm_peak = get_last_mode(values)
    elif modality.lower() in ["t2w", "largest'"]:
        wm_peak = get_largest_mode(values)
    elif modality.lower() in ["md", "first"]:
        wm_peak = get_first_mode(voi)
    else:
        raise ValueError("Invalid modality specified.")
    normalized = (arr / wm_peak) * norm_value
    return normalized

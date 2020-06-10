# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that defines common tools to process histograms in the context of
MRI.
"""

# Imports
import numpy as np
from scipy.signal import argrelmax
import statsmodels.api as sm


def get_largest_mode(arr):
    """ Computes the largest (reliable) peak in the histogram.

    Parameters
    ----------
    arr: array
        the input data.

    Returns
    -------
    largest_peak: int
        index of the largest peak.
    """
    grid, pdf = smooth_hist(arr)
    largest_peak = grid[np.argmax(pdf)]
    return largest_peak


def get_last_mode(arr, remove_tail=True, remove_fraction=0.05):
    """ Computes the last (reliable) peak in the histogram.

    Parameters
    ----------
    arr: array
        the input data.
    remove_tail: bool, default True
        remove rare portions of histogram.
    remove_fraction: float, default 0.05
        discared the specified proportion of hist.

    Returns
    -------
    last_peak: int
        index of the last peak.
    """
    if remove_tail:
        rare_thresh = np.percentile(arr, 1 - remove_fraction)
        which_rare = (arr >= rare_thresh)
        values = arr[which_rare != 1]
    grid, pdf = smooth_hist(values)
    maxima = argrelmax(pdf)[0]
    last_peak = grid[maxima[-1]]
    return last_peak


def get_first_mode(arr, remove_tail=True, remove_fraction=0.01):
    """ Computes the first (reliable) peak in the histogram.

    Parameters
    ----------
    arr: array
        the input data.
    remove_tail: bool, default True
        remove rare portions of histogram.
    remove_fraction: float, default 0.01
        discared the specified proportion of hist.

    Returns
    -------
    first_peak: int
        index of the first peak.
    """
    if remove_tail:
        rare_thresh = np.percentile(arr, 1 - remove_fraction)
        which_rare = (arr >= rare_thresh)
        values = arr[which_rare != 1]
    grid, pdf = smooth_hist(values)
    maxima = argrelmax(pdf)[0]
    first_peak = grid[maxima[0]]
    return first_peak


def smooth_hist(arr):
    """ Use KDE to get smooth estimate of the histogram.

    Parameters
    ----------
    arr: array
        the input data.

    Returns
    -------
    grid: array
        domain of the PDF.
    pdf: array
        kernel density estimate of the PDF.
    """
    values = arr.flatten().astype(np.float64)
    bw = arr.max() / 80
    kde = sm.nonparametric.KDEUnivariate(values)
    kde.fit(kernel="gau", bw=bw, gridsize=80, fft=True)
    pdf = 100. * kde.density
    grid = kde.support
    return grid, pdf

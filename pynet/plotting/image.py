# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Import
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def plot_data(data, slice_axis=2):
    """ Plot an image associated data.

    Currently support 2D or 3D dataset of the form (samples, channels, dim).

    Parameters
    ----------
    data: array (samples, channels, dim)
        the data to be displayed.
    slice_axis: int (optional, default 2)
        the slice axis for 3D data.
    """
    # Check input parameters
    if data.ndim not in range(4, 6):
        raise ValueError("Unsupported data dimension.")

    # Reorganize 3D data
    if data.ndim == 5:
        indices = [0, 1, 2]
        assert slice_axis in indices
        indices.remove(slice_axis)
        indices  = [slice_axis + 1, 0, indices[0] + 1, indices[1] + 1]
        slices = [img.transpose(indices) for img in data]
        data = np.concatenate(slices, axis=0)

    # Plot data on grid
    _data = torchvision.utils.make_grid(torch.from_numpy(data))
    _data = _data.numpy()
    plt.imshow(np.transpose(_data, (1, 2, 0)))


def rescale_intensity(arr, in_range, out_range):
    """ Return arr after stretching or shrinking its intensity levels.

    Parameters
    ----------
    arr: array
        input array.
    in_range, out_range: 2-tuple
        min and max intensity values of input and output arr.

    Returns
    -------
    out: array
        array after rescaling its intensity.
    """
    imin, imax = in_range
    omin, omax = out_range
    out = np.clip(arr, imin, imax)
    out = (out - imin) / float(imax - imin)
    return out * (omax - omin) + omin

# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Import
import numpy as np
from pyqtgraph.Qt import QtGui
import pyqtgraph


def plot_data(data, extradata=None, scroll_axis=2):
    """ Plot an image associated data.
    Currently support on 1D, 2D or 3D data.

    Parameters
    ----------
    data: array
        the data to be displayed.
    extradata: list of array
        if specified concatenate this array with the input data.
    scroll_axis: int (optional, default 2)
        the scroll axis for 3D data.
    """
    # Check input parameters
    if data.ndim not in range(1, 4):
        raise ValueError("Unsupported data dimension.")

    # Concatenate
    if extradata is not None:
        concat_axis = 0 if scroll_axis != 0 else 1
        extradata = [
            rescale_intensity(
                arr=_data,
                in_range=(_data.min(), _data.max()),
                out_range=(data.min(), data.max()))
            for _data in extradata]
        data = np.concatenate([data] + extradata, axis=concat_axis)

    # Create application
    app = pyqtgraph.mkQApp()

    # Create the widget
    if data.ndim == 3:
        indices = [i for i in range(3) if i != scroll_axis]
        indices = [scroll_axis] + indices
        widget = pyqtgraph.image(np.transpose(data, indices))
    elif data.ndim == 2:
        widget = pyqtgraph.image(data)
    else:
        widget = pyqtgraph.plot(data)

    # Run application
    app.exec_()


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

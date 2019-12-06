# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to display signals.
"""

# Import
import numpy as np
import matplotlib.pyplot as plt


def _trim_axs(axs, size):
    """ Little helper to massage the axs list to have correct length...
    """
    axs = axs.flat
    for ax in axs[size:]:
        ax.remove()
    return axs[:size]


def plot_history(history):
    """ Plot an history.

    Parameters
    ----------
    history: pynet History
        the history to be displayed.
    """
    nb_plots = len(history.metrics)
    cols = 3
    rows, rest = divmod(nb_plots, cols)
    if rest > 0:
        rows += 1
    fig, axs = plt.subplots(rows, cols)
    axs = _trim_axs(axs, nb_plots)
    for ax, case in zip(axs, history.metrics):
        ax.set_title(case)
        ax.plot(history[case][1])

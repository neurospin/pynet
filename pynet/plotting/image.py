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


def plot_data(data, slice_axis=2, nb_samples=5, labels=None):
    """ Plot an image associated data.

    Currently support 2D or 3D dataset of the form (samples, channels, dim).

    Parameters
    ----------
    data: array (samples, channels, dim)
        the data to be displayed.
    slice_axis: int, default 2
        the slice axis for 3D data.
    nb_samples: int, default 5
        the number of samples to be displayed.
    labels: list of str, default None
        the data labels to be displayed.
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
    #plt.figure()
    #_data = torchvision.utils.make_grid(torch.from_numpy(data))
    #_data = _data.numpy()
    #plt.imshow(np.transpose(_data, (1, 2, 0)))
    indices = np.random.randint(0, data.shape[0], nb_samples)
    nb_channels = data.shape[1]
    plt.figure(figsize=(15, 7), dpi=200)
    for cnt1, ind in enumerate(indices):
        for cnt2 in range(nb_channels):
            im = data[ind, cnt2]
            plt.subplot(nb_channels, nb_samples, nb_samples * cnt2 + cnt1 + 1)
            plt.axis("off")
            if labels is None:
                plt.title("Image " + str(ind))
            else:
                plt.title(labels[ind])
            plt.imshow(im, cmap="gray")


def plot_segmentation_data(data, mask, nb_samples=5):
    """ Display 'nb_samples' images and segmentation masks stored in data and
    mask.
    Parameters
    ----------
    data: array (samples, channels, dim)
        the data to be displayed.
    mask: array (samples, channels, dim)
        the mask data to be overlayed.
    nb_samples: int, default 5
        the number of samples to be displayed.
    """
    indices = np.random.randint(0, data.shape[0], nb_samples)
    plt.figure(figsize=(15, 7), dpi=200)
    for cnt, ind in enumerate(indices):
        im = data[ind, 0]
        plt.subplot(2, nb_samples, cnt + 1)
        plt.axis("off")
        plt.title("Image " + str(ind))
        plt.imshow(im, cmap="gray")
        mask_im = mask[ind]
        plt.subplot(2, nb_samples, cnt + 1 + nb_samples)
        plt.axis("off")
        plt.imshow(im, cmap="gray")
        plt.imshow(np.argmax(mask_im, axis=0), cmap="jet", alpha=0.3)


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

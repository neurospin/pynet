# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to display images.
"""

# Import
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def plot_data(data, slice_axis=2, nb_samples=5, channel=0, labels=None,
              random=True, rgb=False):
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
    channel: int, default 0
        will select slices with data using the provided channel.
    labels: list of str, default None
        the data labels to be displayed.
    random: bool, default True
        select randomly 'nb_samples' data, otherwise the 'nb_samples' firsts.
    rgb: bool, default False
        if set expect three RGB channels.
    """
    # Check input parameters
    if data.ndim not in range(4, 6):
        raise ValueError("Unsupported data dimension.")
    nb_channels = data.shape[1]
    if rgb:
        if nb_channels != 3:
            raise ValueError("With RGB mode activated expect exactly 3 "
                             "channels.")
        else:
            nb_channels = 1

    # Reorganize 3D data
    if data.ndim == 5:
        indices = [0, 1, 2]
        assert slice_axis in indices
        indices.remove(slice_axis)
        indices = [slice_axis + 1, 0, indices[0] + 1, indices[1] + 1]
        slices = [img.transpose(indices) for img in data]
        data = np.concatenate(slices, axis=0)
    valid_indices = [
        idx for idx in range(len(data)) if data[idx, channel].max() > 0]

    # Plot data on grid
    # plt.figure()
    # _data = torchvision.utils.make_grid(torch.from_numpy(data))
    # _data = _data.numpy()
    # plt.imshow(np.transpose(_data, (1, 2, 0)))
    if random:
        indices = np.random.randint(0, len(valid_indices), nb_samples)
    else:
        if len(valid_indices) < nb_samples:
            nb_samples = len(valid_indices)
        indices = range(nb_samples)
    plt.figure(figsize=(15, 7), dpi=200)
    for cnt1, ind in enumerate(indices):
        ind = valid_indices[ind]
        for cnt2 in range(nb_channels):
            if rgb:
                im = data[ind].transpose(1, 2, 0)
                cmap = None
            else:
                im = data[ind, cnt2]
                cmap = "gray"
            plt.subplot(nb_channels, nb_samples, nb_samples * cnt2 + cnt1 + 1)
            plt.axis("off")
            if cnt2 == 0 and labels is None:
                plt.title("Image " + str(ind))
            elif cnt2 == 0:
                plt.title(labels[ind])
            plt.imshow(im, cmap=cmap)


def plot_segmentation_data(data, mask, slice_axis=2, nb_samples=5):
    """ Display 'nb_samples' images and segmentation masks stored in data and
    mask.

    Currently support 2D or 3D dataset of the form (samples, channels, dim).

    Parameters
    ----------
    data: array (samples, channels, dim)
        the data to be displayed.
    mask: array (samples, channels, dim)
        the mask data to be overlayed.
    slice_axis: int, default 2
        the slice axis for 3D data.
    nb_samples: int, default 5
        the number of samples to be displayed.
    """
    # Check input parameters
    if data.ndim not in range(4, 6):
        raise ValueError("Unsupported data dimension.")

    # Reorganize 3D data
    if data.ndim == 5:
        indices = [0, 1, 2]
        assert slice_axis in indices
        indices.remove(slice_axis)
        indices = [slice_axis + 1, 0, indices[0] + 1, indices[1] + 1]
        slices = [img.transpose(indices) for img in data]
        data = np.concatenate(slices, axis=0)
        slices = [img.transpose(indices) for img in mask]
        mask = np.concatenate(slices, axis=0)
    mask = np.argmax(mask, axis=1)
    valid_indices = [idx for idx in range(len(mask)) if mask[idx].max() > 0]
    print(mask.shape, len(valid_indices))

    # Plot data on grid
    indices = np.random.randint(0, len(valid_indices), nb_samples)
    plt.figure(figsize=(15, 7), dpi=200)
    for cnt, ind in enumerate(indices):
        ind = valid_indices[ind]
        im = data[ind, 0]
        plt.subplot(2, nb_samples, cnt + 1)
        plt.axis("off")
        # plt.title("Image " + str(ind))
        plt.imshow(im, cmap="gray")
        mask_im = mask[ind]
        plt.subplot(2, nb_samples, cnt + 1 + nb_samples)
        plt.axis("off")
        plt.imshow(mask_im, cmap="jet")
        plt.imshow(im, cmap="gray", alpha=0.4)


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

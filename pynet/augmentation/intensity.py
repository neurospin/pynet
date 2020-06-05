# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Common functions to change image intensities.
Code: https://github.com/fepegar/torchio
"""

# Import
import numbers
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
from .transform import compose
from .transform import affine_flow
from .utils import interval


def add_blur(arr, std):
    """ Add random blur using a Gaussian filter.

    Parameters
    ----------
    arr: array
        the input data.
    std: float or 2-uplet
        the standard deviation for Gaussian kernel.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    std = interval(std, lower=0)
    std_random = np.random.uniform(low=std[0], high=std[1], size=1)[0]
    return gaussian_filter(arr, std_random)


def add_noise(arr, mean, std):
    """ Add random Gaussian noise.

    Parameters
    ----------
    arr: array
        the input data.
    mean: float or 2-uplet
        the mean for the Gaussian distribution.
    std: float or 2-uplet
        the standard deviation for the Gaussian distribution.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    std = interval(std, lower=0)
    std_random = np.random.uniform(low=std[0], high=std[1], size=1)[0]
    mean = interval(mean, lower=0)
    mean_random = np.random.uniform(low=mean[0], high=mean[1], size=1)[0]
    noise = np.random.normal(mean_random, std_random, arr.shape)
    return arr + noise


def add_ghosting(arr, axis, n_ghosts=10, intensity=1):
    """ Add random MRI ghosting artifact.

    Parameters
    ----------
    arr: array
        the input data.
    axis: int
        the axis along which the ghosts artifact will be created.
    n_ghosts: int or 2-uplet, default 10
        the number of ghosts in the image. Larger values generate more
        distorted images.
    intensity: float or list of float, default 1
        a number between 0 and 1 representing the artifact strength. Larger
        values generate more distorted images.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    # Leave first 5% of frequencies untouched.
    n_ghosts = interval(n_ghosts, lower=0)
    intensity = interval(intensity, lower=0)
    n_ghosts_random = np.random.randint(
        low=n_ghosts[0], high=n_ghosts[1], size=1)[0]
    intensity_random = np.random.uniform(
        low=intensity[0], high=intensity[1], size=1)[0]
    percentage_to_avoid = 0.05
    values = arr.copy()
    slc = [slice(None)] * len(arr.shape)
    for slice_idx in range(values.shape[axis]):
        slc[axis] = slice_idx
        slice_arr = values[tuple(slc)]
        spectrum = np.fft.fftshift(np.fft.fftn(slice_arr))
        for row_idx, row in enumerate(spectrum):
            if row_idx % n_ghosts_random != 0:
                continue
            progress = row_idx / arr.shape[0]
            if np.abs(progress - 0.5) < (percentage_to_avoid / 2):
                continue
            row *= (1 - intensity_random)
        slice_arr *= 0
        slice_arr += np.abs(np.fft.ifftn(np.fft.ifftshift(spectrum)))
    return values


def add_spike(arr, n_spikes=1, intensity=(0.1, 1)):
    """ Add random MRI spike artifacts.

    Parameters
    ----------
    arr: array
        the input data.
    n_spikes: int, default 1
        the number of spikes presnet in k-space. Larger values generate more
        distorted images.
    intensity: float or 2-uplet, default (0.1, 1)
        Ratio between the spike intensity and the maximum of the spectrum.
        Larger values generate more distorted images.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    intensity = interval(intensity, lower=0)
    spikes_positions = np.random.rand(n_spikes)
    intensity_factor = np.random.uniform(
        low=intensity[0], high=intensity[1], size=1)[0]
    spectrum = np.fft.fftshift(np.fft.fftn(arr)).ravel()
    indices = (spikes_positions * len(spectrum)).round().astype(int)
    for index in indices:
        spectrum[index] = spectrum.max() * intensity_factor
    spectrum = spectrum.reshape(arr.shape)
    result = np.abs(np.fft.ifftn(np.fft.ifftshift(spectrum)))
    return result.astype(np.float32)


def add_biasfield(arr, coefficients=0.5, order=3):
    """ Add random MRI bias field artifact.

    Parameters
    ----------
    arr: array
        the input data.
    coefficients: float, default 0.5
        the magnitude of polynomial coefficients.
    order: int, default 3
        the order of the basis polynomial functions.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    coefficients = interval(coefficients)
    shape = np.array(arr.shape)
    ranges = [np.arange(-size, size) for size in (shape / 2.)]
    bias_field = np.zeros(shape)
    x_mesh, y_mesh, z_mesh = np.asarray(np.meshgrid(*ranges))
    x_mesh /= x_mesh.max()
    y_mesh /= y_mesh.max()
    z_mesh /= z_mesh.max()
    cnt = 0
    for x_order in range(order + 1):
        for y_order in range(order + 1 - x_order):
            for z_order in range(order + 1 - (x_order + y_order)):
                random_coefficient = np.random.uniform(
                    low=coefficients[0], high=coefficients[1], size=1)[0]
                new_map = (
                    random_coefficient * x_mesh ** x_order * y_mesh ** y_order
                    * z_mesh ** z_order)
                bias_field += new_map.transpose(1, 0, 2)
                cnt += 1
    bias_field = np.exp(bias_field).astype(np.float32)
    return arr * bias_field


def add_motion(arr, rotation=10, translation=10, n_transforms=2,
               perturbation=0.3):
    """ Add random MRI motion artifact on the last axis.

    Reference: Shaw et al., 2019, MRI k-Space Motion Artefact Augmentation:
    Model Robustness and Task-Specific Uncertainty.

    Parameters
    ----------
    arr: array
        the input data.
    rotation: float or 2-uplet, default 10
        the rotation in degrees of the simulated movements. Larger
        values generate more distorted images.
    translation: floatt or 2-uplet, default 10
        the translation in voxel of the simulated movements. Larger
        values generate more distorted images.
    n_transforms: int, default 2
        the number of simulated movements. Larger values generate more
        distorted images.
    perturbation: float, default 0.3
        control the intervals between movements. If perturbation is 0, time
        intervals between movements are constant.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    rotation = interval(rotation)
    translation = interval(translation)
    step = 1. / (n_transforms + 1)
    times = np.arange(0, 1, step)[1:]
    shape = arr.shape
    noise = np.random.uniform(
        low=(-step * perturbation), high=(step * perturbation),
        size=n_transforms)
    times += noise
    arrays = [arr]
    for cnt in range(n_transforms):
        random_rotations = np.random.uniform(
            low=rotation[0], high=rotation[1], size=arr.ndim)
        random_translations = np.random.uniform(
            low=translation[0], high=translation[1], size=arr.ndim)
        random_rotations = Rotation.from_euler(
            "xyz", random_rotations, degrees=True)
        random_rotations = random_rotations.as_dcm()
        zoom = [1, 1, 1]
        affine = compose(random_translations, random_rotations, zoom)
        flow = affine_flow(affine, shape)
        locs = flow.reshape(len(shape), -1)
        transformed = map_coordinates(arr, locs, order=3, cval=0)
        arrays.append(transformed.reshape(shape))
    spectra = [np.fft.fftshift(np.fft.fftn(array)) for array in arrays]
    n_spectra = len(spectra)
    if np.any(times > 0.5):
        index = np.where(times > 0.5)[0].min()
    else:
        index = n_spectra - 1
    spectra[0], spectra[index] = spectra[index], spectra[0]
    result_spectrum = np.empty_like(spectra[0])
    last_index = result_spectrum.shape[2]
    indices = (last_index * times).astype(int).tolist()
    indices.append(last_index)
    start = 0
    for spectrum, end in zip(spectra, indices):
        result_spectrum[..., start: end] = spectrum[..., start: end]
        start = end
    result_image = np.abs(np.fft.ifftn(np.fft.ifftshift(result_spectrum)))
    return result_image

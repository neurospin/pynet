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


def add_offset(arr, factor, seed=None):
    """ Add a random intensity offset (shift and scale).

    Parameters
    ----------
    arr: array
        the input data.
    factor: float or 2-uplet
        the offset scale factor [0, 1] for the standard deviation and the mean.
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    factor = interval(factor, lower=factor)
    sigma = interval(factor[0], lower=0)
    mean = interval(factor[1])
    np.random.seed(seed)
    sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
    np.random.seed(seed)
    mean_random = np.random.uniform(low=mean[0], high=mean[1], size=1)[0]
    np.random.seed(seed)
    offset = np.random.normal(mean_random, sigma_random, arr.shape)
    offset += 1
    transformed = arr * offset
    return transformed


def add_blur(arr, snr=None, sigma=None, seed=None):
    """ Add random blur using a Gaussian filter.

    Parameters
    ----------
    arr: array
        the input data.
    snr: float, default None
        the desired signal-to noise ratio used to infer the standard deviation
        for the noise distribution.
    sigma: float or 2-uplet
        the standard deviation for Gaussian kernel.
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    if snr is None and sigma is None:
        raise ValueError("You must define either the desired signal-to noise "
                         "ratio or the standard deviation for the noise "
                         "distribution.")
    if snr is not None:
        s0 = np.max(arr)
        sigma = s0 / snr
    sigma = interval(sigma, lower=0)
    np.random.seed(seed)
    sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
    return gaussian_filter(arr, sigma_random)


def add_noise(arr, snr=None, sigma=None, noise_type="gaussian", seed=None):
    """ Add random Gaussian or Rician noise.

    The noise level can be specified directly by setting the standard
    deviation or the desired signal-to-noise ratio for the Gaussian
    distribution. In the case of Rician noise sigma is the standard deviation
    of the two Gaussian distributions forming the real and imaginary
    components of the Rician noise distribution.

    In anatomical scans, CNR values for GW/WM ranged from 5 to 20 (1.5T and
    3T) for SNR around 40-100 (http://www.pallier.org/pdfs/snr-in-mri.pdf).

    Parameters
    ----------
    arr: array
        the input data.
    snr: float, default None
        the desired signal-to noise ratio used to infer the standard deviation
        for the noise distribution.
    sigma: float or 2-uplet, default None
        the standard deviation for the noise distribution.
    noise_type: str, default 'gaussian'
        the distribution of added noise - can be either 'gaussian' for
        Gaussian distributed noise, or 'rician' for Rice-distributed noise.
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    if snr is None and sigma is None:
        raise ValueError("You must define either the desired signal-to noise "
                         "ratio or the standard deviation for the noise "
                         "distribution.")
    if snr is not None:
        s0 = np.max(arr)
        sigma = s0 / snr
    sigma = interval(sigma, lower=0)
    np.random.seed(seed)
    sigma_random = np.random.uniform(low=sigma[0], high=sigma[1], size=1)[0]
    np.random.seed(seed)
    noise = np.random.normal(0, sigma_random, [2] + list(arr.shape))
    if noise_type == "gaussian":
        transformed = arr + noise[0]
    elif noise_type == "rician":
        transformed = np.square(arr + noise[0])
        transformed += np.square(noise[1])
        transformed = np.sqrt(transformed)
    else:
        raise ValueError("Unsupported noise type.")
    return transformed


def add_ghosting(arr, axis, n_ghosts=10, intensity=1, seed=None):
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
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    # Leave first 5% of frequencies untouched.
    n_ghosts = interval(n_ghosts, lower=0)
    intensity = interval(intensity, lower=0)
    np.random.seed(seed)
    n_ghosts_random = np.random.randint(
        low=n_ghosts[0], high=n_ghosts[1], size=1)[0]
    np.random.seed(seed)
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


def add_spike(arr, n_spikes=1, intensity=(0.1, 1), seed=None):
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
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    intensity = interval(intensity, lower=0)
    np.random.seed(seed)
    spikes_positions = np.random.rand(n_spikes)
    np.random.seed(seed)
    intensity_factor = np.random.uniform(
        low=intensity[0], high=intensity[1], size=1)[0]
    spectrum = np.fft.fftshift(np.fft.fftn(arr)).ravel()
    indices = (spikes_positions * len(spectrum)).round().astype(int)
    for index in indices:
        spectrum[index] = spectrum.max() * intensity_factor
    spectrum = spectrum.reshape(arr.shape)
    result = np.abs(np.fft.ifftn(np.fft.ifftshift(spectrum)))
    return result.astype(np.float32)


def add_biasfield(arr, coefficients=0.5, order=3, seed=None):
    """ Add random MRI bias field artifact.

    Parameters
    ----------
    arr: array
        the input data.
    coefficients: float, default 0.5
        the magnitude of polynomial coefficients.
    order: int, default 3
        the order of the basis polynomial functions.
    seed: int, default None
        seed to control random number generator.

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
    np.random.seed(seed)
    random_coefficients = np.random.uniform(
        low=coefficients[0], high=coefficients[1], size=(order + 1)**3)
    for x_order in range(order + 1):
        for y_order in range(order + 1 - x_order):
            for z_order in range(order + 1 - (x_order + y_order)):
                random_coefficient = random_coefficients[cnt]
                new_map = (
                    random_coefficient * x_mesh ** x_order * y_mesh ** y_order
                    * z_mesh ** z_order)
                bias_field += new_map.transpose(1, 0, 2)
                cnt += 1
    bias_field = np.exp(bias_field).astype(np.float32)
    return arr * bias_field


def add_motion(arr, rotation=10, translation=10, n_transforms=2,
               perturbation=0.3, axis=None, seed=None):
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
    axis: int, default None
        the k-space filling axis. If not specified, randomize the k-space
        filling axis.
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    rotation = interval(rotation)
    translation = interval(translation)
    if axis is None:
        np.random.seed(seed)
        axis = np.random.randint(low=0, high=arr.ndim, size=1)[0]
    step = 1. / (n_transforms + 1)
    times = np.arange(0, 1, step)[1:]
    shape = arr.shape
    noise = np.random.uniform(
        low=(-step * perturbation), high=(step * perturbation),
        size=n_transforms)
    times += noise
    arrays = [arr]
    np.random.seed(seed)
    random_rotations = np.random.uniform(
        low=rotation[0], high=rotation[1], size=(n_transforms, arr.ndim))
    np.random.seed(seed)
    random_translations = np.random.uniform(
        low=translation[0], high=translation[1], size=(n_transforms, arr.ndim))
    for cnt in range(n_transforms):
        random_rotations = Rotation.from_euler(
            "xyz", random_rotations[cnt], degrees=True)
        random_rotations = random_rotations.as_dcm()
        zoom = [1, 1, 1]
        affine = compose(random_translations[cnt], random_rotations, zoom)
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
    slc = [slice(None)] * arr.ndim
    slice_size = result_spectrum.shape[axis]
    indices = (slice_size * times).astype(int).tolist()
    indices.append(slice_size)
    start = 0
    for spectrum, end in zip(spectra, indices):
        slc[axis] = slice(start, end)
        result_spectrum[tuple(slc)] = spectrum[tuple(slc)]
        start = end
    result_image = np.abs(np.fft.ifftn(np.fft.ifftshift(result_spectrum)))
    return result_image

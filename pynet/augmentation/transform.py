# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that defines common tools to deal with spatial transformations.
Code: https://github.com/matthew-brett/transforms3d
Code: https://github.com/bsciolla/gaussian-random-fields
"""

# Import
import math
import numpy as np
import scipy.fftpack


# Caching dictionary for common shear Ns, indices
_shearers = {}
for n in range(1, 11):
    x = (n**2 + n) / 2.0
    i = n + 1
    _shearers[x] = (i, np.triu(np.ones((i, i)), 1).astype(bool))


def affine_flow(affine, shape):
    """ Generates a flow field given an affine matrix.

    Parameters
    ----------
    affine: Tensor (4, 4)
        an affine transform.
    shape: tuple
        the target output image shape.

    Returns
    -------
    flow: Tensor
        the generated affine flow field.
    """
    ranges = [np.arange(size) for size in shape]
    ranges[0], ranges[1] = ranges[1], ranges[0]
    mesh = np.asarray(np.meshgrid(*ranges))
    mesh[[0, 1]] = mesh[[1, 0]]
    locs = np.asarray(mesh)
    offset = (np.array(shape) - 1) / 2
    locs = np.transpose(locs.T - offset)
    ones = np.ones([1] + list(locs.shape)[1:], dtype=locs.dtype)
    homography_grid = np.concatenate([locs, ones], axis=0)
    shape = homography_grid.shape
    homography_grid = homography_grid.reshape(shape[0], -1)
    flow = np.dot(affine, homography_grid)
    flow = flow.reshape(shape)
    flow = flow[:(len(shape) - 1)]
    flow = np.transpose(flow.T + offset)
    return flow


def compose(T, R, Z, S=None):
    """ Compose translations, rotations, zooms, [shears]  to affine

    Parameters
    ----------
    T: array (N,)
        translations, where N is usually 3 (3D case)
    R: array (N, N)
        rotation matrix where N is usually 3 (3D case)
    Z: array (N,)
        zooms, where N is usually 3 (3D case)
    S: array (P,), default None
        shear vector, such that shears fill upper triangle above
        diagonal to form shear matrix.  P is the (N-2)th Triangular
        number, which happens to be 3 for a 4x4 affine (3D case)

    Returns
    -------
    A: array (N+1, N+1)
        affine transformation matrix where N usually == 3
        (3D case)
    """
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError("Wrong rotation matrix shape.")
    A = np.eye(n + 1)
    if S is not None:
        Smat = striu2mat(S)
        ZS = np.dot(np.diag(Z), Smat)
    else:
        ZS = np.diag(Z)
    A[:n, :n] = np.dot(R, ZS)
    A[:n, n] = T[:]
    return A


def striu2mat(striu):
    """ Construct shear matrix from upper triangular vector.

    Parameters
    ----------
    striu: array (N,)
       vector giving triangle above diagonal of shear matrix.

    Returns
    -------
    SM: array (N, N)
       shear matrix.
    """
    n = len(striu)
    if n in _shearers:
        N, inds = _shearers[n]
    else:
        N = ((-1 + math.sqrt(8 * n + 1)) / 2.0) + 1  # n+1 th root
        if N != math.floor(N):
            raise ValueError(
                "{0} is a strange number of shear elements".format(n))
        N = int(N)
        inds = np.triu(np.ones((N, N)), 1).astype(bool)
    M = np.eye(N)
    M[inds] = striu
    return M


def fftind(shape):
    """ Returns a numpy array of shifted Fourier coordinates.

    Parameters
    ----------
    shape: uplet
        the shape of the coordinate array to create.

    Returns
    -------
    k_ind: array (2, size, size) with:
        shifted Fourier coordinates.
    """
    half_shape = (np.array(shape) + 1) / 2
    k_ind = np.mgrid[[slice(0, size) for size in shape]]
    k_ind = np.transpose(k_ind.T - half_shape)
    k_ind = scipy.fftpack.fftshift(k_ind)
    return k_ind


def gaussian_random_field(shape, alpha=3.0, normalize=True, seed=None):
    """ Generates 3D gaussian random maps.
    The probability distribution of each variable follows a Normal
    distribution.

    Parameters
    ----------
    shape: uplet,
        the shape of the output Gaussian random fields.
    alpha: flaot, default 3
        the power of the power-law momentum distribution.
    normalize: bool, default True
        normalizes the Gaussian field to have an average of 0.0 and a standard
        deviation of 1.0.
    seed: int, default None
        seed to control random number generator.

    Returns
    -------
    gfield: array
        the gaussian random field.
    """
    # Defines momentum indices
    k_idx = fftind(shape)

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(np.sum(k_idx**2, axis=0) + 1e-10, -alpha/4.0)
    amplitude[0, 0] = 0

    # Draws a complex gaussian random noise with normal (circular) distribution
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(size=shape) + 0j
    if seed is not None:
        np.random.seed(seed + 1)
    noise += 1j * np.random.normal(size=shape)

    # To real space
    gfield = np.fft.ifft2(noise * amplitude).real

    # Sets the standard deviation to one
    if normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield / np.std(gfield)

    return gfield

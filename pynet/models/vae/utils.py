# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module containing VAE utilities.

Code: https://github.com/YannDubs/disentangling-vae
"""

# Imports
import os
from operator import itemgetter
import numpy as np
from scipy import stats
from PIL import Image, ImageDraw
import torch
from torch.distributions import Normal, kl_divergence
from torchvision.utils import make_grid


def get_traversal_range(mean=0, std=1, max_traversal=0.475):
    """ Return the corresponding traversal range in absolute terms.

    Parameters
    ----------
    mean: float, default 0
        normal distribution mean.
    std: float, default 1
        normal distribution sigma.
    max_traversal: float, default 0.475
        the maximum displacement induced by a latent traversal. Symmetrical
        traversals are assumed. If m >= 0.5 then uses absolute value traversal,
        if m < 0.5 uses a percentage of the distribution (quantile),
        e.g. for the prior the distribution is a standard normal so m = 0.45
        corresponds to an absolute value of 1.645 because 2m = 90%% of a
        standard normal is between -1.645 and 1.645. Note in the case
        of the posterior, the distribution is not standard normal anymore.

    Returns
    -------
    out: 2-uplet
        traversal range.
    """
    if max_traversal < 0.5:
        max_traversal = (1 - 2 * max_traversal) / 2
        max_traversal = stats.norm.ppf(max_traversal, loc=mean, scale=std)
    return (- max_traversal, max_traversal)


def traverse_line(model, idx, n_samples, data=None):
    """ Return latent samples corresponding to a traversal of a latent
    variable indicated by idx.

    Parameters
    ----------
    model: nn.Module
        the trained network.
    idx: int
        index of continuous dimension to traverse. If the continuous latent
        vector is 10 dimensional and idx = 7, then the 7th dimension
        will be traversed while all others are fixed.
    n_samples: int
        number of samples to generate.
    data: torch.Tensor (N, C, H, W), default None
        data to use for computing the posterior. If 'None' then use the
        mean of the prior (all zeros) for all other dimensions.

    Returns
    -------
    samples: torch.Tensor (n_samples, latent_size)
    """
    if data is None:
        samples = torch.zeros(n_samples, model.latent_dim)
        traversals = torch.linspace(*get_traversal_range(), steps=n_samples)
    else:
        if data.size(dim=0) > 1:
            raise ValueError(
                "Every value should be sampled from the same posterior")
        with torch.no_grad():
            posterior = model.encode(data)
            samples = posterior.sample()
            samples = samples.cpu().repeat(n_samples, 1)
        post_mean_idx = posterior.loc.cpu()[0, idx]
        post_std_idx = posterior.scale.cpu()[0, idx]
        traversals = torch.linspace(*get_traversal_range(
            mean=post_mean_idx, std=post_std_idx), steps=n_samples)
    samples[:, idx] = traversals
    return samples


def traversals(model, device, data=None, n_per_latent=8, n_latents=None):
    """ Plot traverse through all latent dimensions (prior or posterior) one
    by one and plots a grid of images where each row corresponds to a latent
    traversal of one latent dimension.

    Parameters
    ----------
    model: nn.Module
        the trained network.
    device: torch.device
        the device.
    data: torch.Tensor (N, C, H, W), default None
        data to use for computing the posterior. If 'None' then use the
        mean of the prior (all zeros) for all other dimensions.
    n_per_latent: int, default 8
        the number of points to include in the traversal of a latent dimension,
        i.e. the number of columns.
    n_latents: int, default None
        the number of latent dimensions to display, i.e. the number of rows.
        If 'None' uses all latents.
    """
    sampling_type = "prior" if data is None else "posterior"
    n_latents = n_latents or model.latent_dim
    size = (n_latents, n_per_latent)
    latent_samples = [traverse_line(model, dim, n_per_latent, data=data)
                      for dim in range(n_latents)]
    latent_samples = torch.cat(latent_samples, dim=0).to(device)
    decoded_traversal = model.p_to_prediction(model.decode(latent_samples))
    n_images, *img_shape = decoded_traversal.shape
    n_rows = n_images // n_per_latent
    decoded_traversal = decoded_traversal.reshape(
        n_rows, n_per_latent, *img_shape)
    return decoded_traversal


def reconstruct_traverse(model, data, n_per_latent=8, n_latents=None,
                         is_posterior=False, filename=None):
    """ Creates a figure whith first row for original images, second are
    reconstructions, rest are traversals (prior or posterior) of the latent
    dimensions.

    Parameters
    ----------
    model: nn.Module
        the trained network.
    data: torch.Tensor (N, C, H, W)
        data to be reconstructed.
    n_per_latent: int, default 8
        the number of points to include in the traversal of a latent
        dimension, i.e. the number of columns.
    n_latents: int, default None
        the number of latent dimensions to display, i.e. the number of rows.
        If 'None' uses all latents.
    is_posterior: bool, default False
        whether to sample from the posterior.
    filename: str, default None
        path to save the finale image.
    """
    device = data.get_device()
    n_latents = n_latents or model.latent_dim
    q = model.encode(data[:n_per_latent])
    dimension_wise_kl_loss = kl_divergence(
        q, Normal(0, 1)).mean(dim=0)[:n_latents]
    reconstruction = model.reconstruct(data[:n_per_latent], sample=False)
    reconstruction = np.expand_dims(reconstruction, axis=0)
    original = data[:n_per_latent].cpu().numpy()
    original = np.expand_dims(original, axis=0)
    traversal = traversals(
        model, device, data=data[:1, ...] if is_posterior else None,
        n_per_latent=n_per_latent, n_latents=n_latents)
    traversal = np.asarray([arr for _, arr in sorted(
        zip(dimension_wise_kl_loss, traversal), key=itemgetter(0))])
    concatenated = np.concatenate(
        (original, reconstruction, traversal), axis=0)
    mosaic = make_mosaic_img(concatenated)
    concatenated = Image.fromarray(mosaic)
    labels = ["orig", "recon"]
    traversal_labels = [
        "dim={0} KL={1:.4f}".format(dim + 1, kl)
        for dim, kl in enumerate(dimension_wise_kl_loss)]
    traversal_labels = [label for _, label in sorted(
        zip(dimension_wise_kl_loss, traversal_labels), key=itemgetter(0))]
    labels += traversal_labels
    concatenated = add_labels(concatenated, labels)
    if filename is not None:
        concatenated.save(filename)
    return concatenated


def add_labels(input_image, labels):
    """ Adds labels next to rows of an image.

    Parameters
    ----------
    input_image: PIL.Image
        the image to which to add the labels.
    labels: list
        the list of labels to plot.
    """
    n_labels = len(labels)
    width, height = (input_image.width, input_image.height)
    new_width = width + 100
    new_size = (new_width, height)
    new_img = Image.new("RGB", new_size, color="white")
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)
    for idx, text in enumerate(labels):
        draw.text(xy=(new_width - 100 + 0.005,
                      int((idx / n_labels + 1 / (2 * n_labels)) * height)),
                  text=text, fill=(0, 0, 0))
    return new_img


def make_mosaic_img(arr):
    """ Converts a grid of image array into a single mosaic.

    Parameters
    ----------
    arr: numpy.ndarray (ROWS, COLS, C, H, W)
        organized images all of the same size to generate the mosaic.
    """
    img_shape = arr.shape[2:]
    nrow = arr.shape[1]
    tensor = torch.from_numpy(arr.reshape(-1, *img_shape))
    grid = make_grid(tensor, nrow=nrow, normalize=True, range=(0, 1),
                     padding=2, pad_value=1, scale_each=True)
    mosaic = grid.mul_(255).clamp_(0, 255).permute(1, 2, 0)
    mosaic = mosaic.to("cpu", torch.uint8).numpy()
    return mosaic

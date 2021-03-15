"""
pynet dense fonctional brain networks extraction
================================================

Credit: A Grigis

Spatiotemporal Attention Autoencoder (STAAE) for ADHD Classification,
MICCAI, 2020.
DEEP VARIATIONAL AUTOENCODER FOR MODELING FUNCTIONAL BRAIN NETWORKS AND ADHD
IDENTIFICATION, ISBI 2020.
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()


# Import
import logging
import numpy as np
import random
import math
import time
import nibabel
import scipy.ndimage as ndimage
from nilearn import datasets
from nilearn import masking
from nilearn.image.resampling import resample_to_img
from nilearn.input_data import MultiNiftiMasker
import pandas as pd
from sklearn.linear_model import Lasso
import pynet
from pynet.datasets import DataManager
from pynet.plotting import Board, update_board
from pynet import NetParameters
from pynet.interfaces import VAENetEncoder, STAAENetEncoder
from pynet.utils import setup_logging
from pynet.interfaces import DeepLearningInterface
import torch
from torch import nn


# Global parameters
MODEL = "STAAE"  # "DVAE"
DATADIR = "/neurospin/nsap/research/stAAE/data"
WORKDIR = "/neurospin/nsap/research/stAAE"
DATAFILE = os.path.join(WORKDIR, "ADHD40.npy")
MASKFILE = os.path.join(WORKDIR, "ADHD40_mask.nii.gz")
SEGFILE = "./MNI152_T1_1mm_Brain_FAST_seg.nii.gz"
STRUCTFILE = "./MNI152_T1_2mm_strucseg_periph.nii.gz"
SEED = 1234
BATCH_SIZE = 200
random.seed(SEED)
np.random.seed(SEED)
setup_logging(level="info")
logger = logging.getLogger("pynet")


# Prepare data
adhd_dataset = datasets.fetch_adhd(n_subjects=40, data_dir=DATADIR)
func_filenames = adhd_dataset.func
print("Functional nifti image: {0}...{1} ({2})".format(
    func_filenames[0], func_filenames[1], len(func_filenames)))

# Build an EPI-based mask because we have no anatomical data
if not os.path.isfile(MASKFILE):
    target_img = nibabel.load(func_filenames[0])
    target_mask = (target_img.get_data()[..., 0] != 0).astype(int)
    template = nibabel.load(SEGFILE)
    struct = nibabel.load(STRUCTFILE)
    resampled_template = resample_to_img(
        template, target_img, interpolation="nearest")
    resampled_struct = resample_to_img(
        struct, target_img, interpolation="nearest")
    mask = (resampled_template.get_data() == 2).astype(float)
    # mask = ndimage.gaussian_filter(mask, sigma=1.25)
    mask = (mask >= 0.3).astype(int)
    mask = mask & resampled_struct.get_data() & target_mask
    mask_img = nibabel.Nifti1Image(mask, target_img.affine)
    nibabel.save(mask_img, MASKFILE)
else:
    mask_img = nibabel.load(MASKFILE)

# Mask and preproc EPI data
# Build an EPI-based mask because we have no anatomical data
# Register atlas to select gray matter
masker = MultiNiftiMasker(
    mask_img=mask_img,
    standardize=True,
    detrend=1,
    smoothing_fwhm=6.)
masker.fit()
if not os.path.isfile(DATAFILE):
    iterator = np.concatenate(masker.transform(func_filenames), axis=0)
    print(iterator.shape)
    np.save(DATAFILE, iterator)
else:
    iterator = np.load(DATAFILE)

# Data iterator
iterator = np.expand_dims(iterator, axis=1)
manager = DataManager.from_numpy(train_inputs=iterator, batch_size=BATCH_SIZE,
                                 add_input=True)

# Create model
if MODEL == "DVAE":
    losses = pynet.get_tools(tool_name="losses")
    loss_klass = losses["BetaHLoss"]
    params = NetParameters(
        input_channels=1,
        input_dim=iterator.shape[-1],
        conv_flts=None,
        dense_hidden_dims=[256, 128, 64],
        latent_dim=32)
    interface = VAENetEncoder(
        params,
        optimizer_name="Adam",
        learning_rate=0.00001,
        loss=loss_klass(use_mse=True, beta=1.),
        use_cuda=False)
    name = MODEL
else:
    params = NetParameters(
        input_dim=iterator.shape[-1])
    interface = STAAENetEncoder(
        params,
        optimizer_name="Adam",
        learning_rate=0.001,
        loss_name="MSELoss",
        use_cuda=True)
    name = MODEL

# Train model
interface.board = Board(
    port=8097, host="http://localhost", env="dvae")
interface.add_observer("after_epoch", update_board)
test_history, train_history = interface.training(
    manager=manager,
    nb_epochs=50,
    checkpointdir=os.path.join(WORKDIR, "checkpoint_" + name),
    fold_index=0,
    with_validation=False)

# Create test data
manager = DataManager.from_numpy(test_inputs=iterator, batch_size=BATCH_SIZE,
                                 add_input=True)


def dummy_loss(*args, **kwargs):
    return -1


# Get latent parameters
interface.model.nodecoding = True
interface.loss = dummy_loss
y_pred, X, y_true, loss, values = interface.testing(
    manager=manager,
    with_logit=False,
    predict=False)


# Compute encoder weights
y = np.squeeze(y_pred)
iterator = np.squeeze(iterator)
print(y.shape, iterator.shape)
clf = Lasso(alpha=0.01)
clf.fit(y, iterator)
print(clf.coef_.shape)


def thresholding(components):
    S = np.sqrt(np.sum(components ** 2, axis=1))
    S[S == 0] = 1
    components /= S[:, np.newaxis]

    # Flip signs in each composant so that positive part is l1 larger
    # than negative part. Empirically this yield more positive looking maps
    # than with setting the max to be positive.
    for component in components:
        if np.sum(component > 0) < np.sum(component < 0):
            component *= -1
    return components


def plot_net(components):
    from nilearn.plotting import plot_prob_atlas
    from nilearn.image import iter_img
    from nilearn.plotting import plot_stat_map, show

    components_img = masker.inverse_transform(components)

    # Plot all ICA components together
    plot_prob_atlas(components_img, title="All ICA components")

    for i, cur_img in enumerate(iter_img(components_img)):
        plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                      cut_coords=10, colorbar=False)

    show()


# Display thresholded FBNs
components = thresholding(clf.coef_.T)
plot_net(components)

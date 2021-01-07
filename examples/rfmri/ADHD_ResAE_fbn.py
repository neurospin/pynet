"""
pynet deep conv fonctional brain networks extraction
====================================================

Credit: A Grigis

Discovering Functional Brain Networks with 3D Residual Autoencoder (ResAE),
MICCAI 2020
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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from nilearn import datasets
from nilearn.input_data import MultiNiftiMasker
from pynet.datasets import DataManager
from pynet.plotting import Board, update_board
from pynet import NetParameters
from pynet.models.vae.vunet import DecodeLoss
from pynet.interfaces import ResAENetEncoder
from pynet.utils import setup_logging
from pynet.interfaces import DeepLearningInterface


import torch
import torch.nn as nn
import torch.nn.functional as func
from functools import partialmethod
from functools import partial


# Global parameters
DATADIR = "/neurospin/nsap/research/resNet/data"
WORKDIR = "/neurospin/nsap/research/resNet"
PLOTDIR = "/neurospin/nsap/research/resNet/fbns"
DATAFILE = os.path.join(WORKDIR, "ADHD40.npy")
PREDFILE = os.path.join(WORKDIR, "ADHD40_pred.npy")
MASKFILE = os.path.join(WORKDIR, "ADHD40_mask.nii.gz")
SEGFILE = "./MNI152_T1_1mm_Brain_FAST_seg.nii.gz"
SEED = 1234
BATCH_SIZE = 20
EPOCH = 99
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
    mask = (target_img.get_data()[..., 0] != 0).astype(int)
    mask_img = nibabel.Nifti1Image(mask, target_img.affine)
    nibabel.save(mask_img, MASKFILE)
else:
    mask_img = nibabel.load(MASKFILE)

# Mask and preproc EPI data
masker = MultiNiftiMasker(
    mask_img=mask_img,
    standardize=True)
masker.fit()
if not os.path.isfile(DATAFILE):
    y = np.concatenate(masker.transform(func_filenames), axis=0)
    print(y.shape)
    np.save(DATAFILE, y)
else:
    y = np.load(DATAFILE)
iterator = masker.inverse_transform(y).get_fdata()
iterator = iterator.transpose((3, 0, 1, 2))
iterator = np.expand_dims(iterator, axis=1)
print(iterator.shape)

# Data iterator
manager = DataManager.from_numpy(train_inputs=iterator, batch_size=BATCH_SIZE,
                                 add_input=True)

# Create model
name = "ResAENet"
model_weights = os.path.join(
    WORKDIR, "checkpoint_" + name, "model_0_epoch_{0}.pth".format(EPOCH))
if os.path.isfile(model_weights):
    pretrained = model_weights
else:
    pretrained = None
params = NetParameters(
    input_shape=(61, 73, 61),
    cardinality=1,
    layers=[3, 4, 6, 3],
    n_channels_in=1,
    decode=True)
interface = ResAENetEncoder(
    params,
    optimizer_name="Adam",
    learning_rate=0.001,
    loss_name="MSELoss",
    pretrained=pretrained,
    use_cuda=True)

# Train model
if pretrained is None:
    interface.board = Board(
        port=8097, host="http://localhost", env="resnet")
    interface.add_observer("after_epoch", update_board)
    test_history, train_history = interface.training(
        manager=manager,
        nb_epochs=100,
        checkpointdir=os.path.join(WORKDIR, "checkpoint_" + name),
        fold_index=0,
        with_validation=False)


def dummy_loss(*args, **kwargs):
    return -1


# Get latent parameters
if not os.path.isfile(PREDFILE):
    manager = DataManager.from_numpy(
        test_inputs=iterator, batch_size=BATCH_SIZE, add_input=True)
    interface.model.decode = False
    interface.loss = dummy_loss
    z_pred, X, y_true, loss, values = interface.testing(
        manager=manager,
        with_logit=False,
        predict=False)
    np.save(PREDFILE, z_pred)
else:
    z_pred = np.load(PREDFILE)
print(z_pred.shape, y.shape)


def get_fbns(X, Z, alpha=0.0005):
    """ Functional Brain Networks (FBNs) Estimation.

    Parameters
    ----------
    X: array (t, n)
        group-wise transformed fMRI data where n is the number of samples
        and t is the group-wise number of timepoints.
    Z: array (t, d)
        temporal features (latent variables) where d is the latent space
        dimension and t is the group-wise number of timepoints..

    Returns
    -------
    W: array (n, d)
        coefficient matrix where each row contains a FBN.
    """
    if alpha == 0:
        estimator = LinearRegression()
    else:
        estimator = Lasso(alpha, tol=0.1, max_iter=100)
    estimator.fit(X, Z)
    W = estimator.coef_
    return W


def thresholding(W, thr=35):
    """ Threshold Functional Brain Networks (FBNs) coefficients.

    Parameters
    ----------
    W: array (n, d)
        coefficient matrix where each row contains a FBN.
    thr: int, default 35
        deffine the threshold in percent from the maximum FBN coefficient.

    Returns
    -------
    W: array (n, d)
        coefficient matrix where each row contains a FBN.
    """
    for idx, component in enumerate(W):
        abs_maps = np.amax(component)
        threshold = thr * abs_maps / 100
        component[component < threshold] = 0
        W[idx, :] = component
    return W


def display_fbns(W, destdir):
    """ Display Functional Brain Networks (FBNs) coefficients.

    Parameters
    ----------
    W: array (n, d)
        coefficient matrix where each row contains a FBN.
    destdir: str
        the destination folder where FBN will be saved.
    """
    from nilearn.plotting import plot_stat_map
    from nilearn.image import iter_img
    project_W = masker.inverse_transform(W)
    if not os.path.exists(destdir):
        os.mkdir(destdir)
    for idx, cur_fbn in enumerate(iter_img(project_W)):
        outfile = os.path.join(destdir, "fbn_{0}.png".format(idx))
        plot_stat_map(
            cur_fbn, bg_img="utils/MNI152_T1_1mm.nii.gz", display_mode="z",
            black_bg=True, annotate=0, colorbar=0, output_file=outfile)


# Compute encoder weights
fbn = get_fbns(y, np.squeeze(z_pred), alpha=0)
print(fbn.shape)

# Display thresholded FBNs
fbn = thresholding(fbn, thr=100)
display_fbns(fbn, PLOTDIR)

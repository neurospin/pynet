"""
pynet data augmentation overview
================================

Credit: A Grigis

pynet contains a set of tools to efficiently augment 3D medical images that
is crutial for deep learning applications. It includes random affine/non linear
transformations, simulation of intensity artifacts due to MRI magnetic field
inhomogeneity or k-space motion artifacts, and others.

Load the data
-------------

We load the Brats dataset and select the first MRI brain image.
"""

import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()

import time
import numpy as np
import nibabel
import random
from pynet.datasets import DataManager, fetch_toy, fetch_brats
from pynet.preprocessing import rescale, downsample

datasetdir = "/tmp/toy"
data = fetch_toy(datasetdir=datasetdir)
image = nibabel.load(data.t1w_path)
image = rescale(downsample(image.get_data(), scale=4), dynamic=(0, 255))

#############################################################################
# Define deformations
# -------------------
#
# We now declare MRI brain deformation functions. The deformation can be
# combined with the Transformer class.

from pynet.augmentation import add_blur
from pynet.augmentation import add_noise
from pynet.augmentation import add_ghosting
from pynet.augmentation import add_spike
from pynet.augmentation import add_biasfield
from pynet.augmentation import add_motion
from pynet.augmentation import add_offset
from pynet.augmentation import flip
from pynet.augmentation import affine
from pynet.augmentation import deformation
from pynet.augmentation import Transformer

compose_transforms = Transformer(with_channel=False)
compose_transforms.register(
    flip, probability=0.5, axis=0, apply_to=["all"])
compose_transforms.register(
    add_blur, probability=1, sigma=4, apply_to=["all"])
transforms = {
    "add_blur": (add_blur, {"sigma": 4}),
    "add_noise": (add_noise, {"snr": 5., "noise_type": "rician"}),
    "flip": (flip, {"axis": 0}),
    "affine": (affine, {"rotation": 5, "translation": 0, "zoom": 0.05}),
    "add_ghosting": (add_ghosting, {"n_ghosts": (4, 10), "axis": 2,
                                   "intensity": (0.5, 1)}),
    "add_spike": (add_spike, {"n_spikes": 1, "intensity": (0.1, 1)}),
    "add_biasfield": (add_biasfield, {"coefficients": 0.5}),
    "deformation": (deformation, {"max_displacement": 4, "alpha": 3}),
    "add_motion": (add_motion, {"rotation": 10, "translation": 10,
                                "n_transforms": 2, "perturbation": 0.3}),
    "add_offset": (add_offset, {"factor": (0.05, 0.1)}),
    "compose_transforms": (compose_transforms, {}),
}

#############################################################################
# Test transformations
# --------------------
#
# We now apply the transformations on the loaded image. Results are
# directly displayed in your browser at http://localhost:8097.

from pynet.plotting import Board

board = Board(port=8097, host="http://localhost", env="data-augmentation")
for cnt in range(10):
    print("Iteration: ", cnt)
    for key, (fct, kwargs) in transforms.items():
        images = np.asarray([image, np.clip(fct(image, **kwargs), 0, 255)])
        images = images[..., images.shape[-1] // 2]
        images = np.expand_dims(images, axis=1)
        board.viewer.images(
            images, opts={"title": key, "caption": key}, win=key)
    time.sleep(1)


#############################################################################
# Data augmentation
# -----------------
#
# We now illustrate how we can use the Transformer in combinaison with
# the DataManager to perform data augmentation during training. Results are
# directly displayed in your browser at http://localhost:8097.

datasetdir = "/neurospin/nsap/processed/deepbrain/tumor/data/brats"
data = fetch_brats(datasetdir=datasetdir)

board = Board(port=8097, host="http://localhost", env="data-augmentation")
compose_transforms = Transformer()
compose_transforms.register(
    flip, probability=0.5, axis=0, apply_to=["input", "output"])
compose_transforms.register(
    add_blur, probability=1, sigma=4, apply_to=["input"])
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    output_path=data.output_path,
    number_of_folds=2,
    batch_size=2,
    test_size=0.1,
    sample_size=0.1,
    sampler=None,
    add_input=True,
    data_augmentation_transforms=[compose_transforms])
loaders = manager.get_dataloader(
    train=True,
    validation=False,
    fold_index=0)
for dataitem in loaders.train:
    print("-" * 50)
    print(dataitem.inputs.shape, dataitem.outputs.shape, dataitem.labels)
    images = [dataitem.inputs[0, 0].numpy(), dataitem.inputs[0, 1].numpy(),
              dataitem.outputs[0, 0].numpy(), dataitem.outputs[0, 1].numpy(),
              dataitem.outputs[0, 4].numpy(), dataitem.outputs[0, 5].numpy()]
    images = np.asarray(images)
    images = np.expand_dims(images, axis=1)
    images = images[..., images.shape[-1] // 2]
    images = rescale(images, dynamic=(0, 255))
    board.viewer.images(
        images, opts={"title": "transformer", "caption": "transformer"},
        win="transformer")
    time.sleep(2)

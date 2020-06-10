"""
pynet data preprocessing
========================

Credit: A Grigis

pynet contains a set of tools to efficiently preprocess 3D medical images that
is crutial for deep learning applications. It includes reorientation, bias
field correction, affine alignement, and intensity normalization.

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
import random
import nibabel
from pynet.datasets import fetch_toy
from pynet.preprocessing import rescale

datasetdir = "/tmp/toy"
data = fetch_toy(datasetdir=datasetdir)
image = nibabel.load(data.t1w_path)
target = nibabel.load(data.template_path)

#############################################################################
# Define preprocessing steps
# --------------------------
#
# We now declare MRI brain preprocessing functions that can be
# combined with the Processor class.

from pynet.preprocessing import zscore_normalize
from pynet.preprocessing import kde_normalize
from pynet.preprocessing import reorient2std
from pynet.preprocessing import biasfield
from pynet.preprocessing import register
from pynet.preprocessing import downsample
from pynet.preprocessing import padd
from pynet.preprocessing import scale


small_image = nibabel.Nifti1Image(
    downsample(image.get_data(), scale=4), image.affine)
processes = {
    "zscore_normalize": (zscore_normalize, {"mask": None}),
    "kde_normalize": (kde_normalize, {
        "mask": None, "modality": "T1w", "norm_value": 1}),
    "reorient2std": (reorient2std, {}),
    "register": (register, {
        "target": small_image, "cost": "corratio", "interp": "trilinear", "dof": 6}),
    "biasfield": (biasfield, {"nb_iterations": 3}),
    "downsample": (downsample, {"scale": 2}),
    "padd": (padd, {"shape": [256, 256, 256], "fill_value": 0}),
}

#############################################################################
# Test preprocessings
# --------------------
#
# We now apply the preprocessing steps on the loaded image. Results are
# directly displayed in your browser at http://localhost:8097.

from pynet.plotting import Board

board = Board(port=8097, host="http://localhost", env="data-preprocessing")
for key, (fct, kwargs) in processes.items():
    print("Processing {0}...".format(key))
    if key in ("reorient2std", "biasfield", "register"):
        normalized = fct(small_image, **kwargs).get_data()
    else:
        normalized = fct(small_image.get_data(), **kwargs)
    if key in ("padd", "downsample", "register"):
        images = np.expand_dims(rescale(normalized, dynamic=(0, 255)), axis=0)
    else:
        images = np.asarray([rescale(small_image.get_data(), dynamic=(0, 255)),
                             rescale(normalized, dynamic=(0, 255))])
    images = images[..., images.shape[-1] // 2]
    images = np.expand_dims(images, axis=1)
    board.viewer.images(
        images, opts={"title": key, "caption": key}, win=key)
    print("Done.")
time.sleep(10)


#############################################################################
# Preprocessing pipeline
# ----------------------
#
# We now illustrate how we can use the Processor to preprocess the MRI
# images. Results are directly displayed in your browser at
# http://localhost:8097.

from pynet.preprocessing import Processor

board = Board(port=8097, host="http://localhost", env="data-preprocessing")
pipeline = Processor()
pipeline.register(reorient2std, apply_to="image")
pipeline.register(scale, scale=2, apply_to="image")
pipeline.register(biasfield, apply_to="image")
pipeline.register(register, target=target, apply_to="image")
pipeline.register(zscore_normalize, apply_to="array")
key = "pipeline"
normalized = pipeline(image)
print(image.shape, target.shape, normalized.shape)
images = np.asarray([rescale(target.get_data(), dynamic=(0, 255)),
                     rescale(normalized.get_data(), dynamic=(0, 255))])
images = images[..., images.shape[-1] // 2]
images = np.expand_dims(images, axis=1)
board.viewer.images(
    images, opts={"title": key, "caption": key}, win=key)

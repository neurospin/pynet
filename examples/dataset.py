"""
pynet dataset helpers overview
==============================

Credit: A Grigis

pynet is a Python package related to deep learning and its application in
MRI mediacal data analysis. It is accessible to everybody, and is reusable
in various contexts. The project is hosted on github:
https://github.com/neurospin/pynet.

First checks
------------

In order to test if the 'pynet' package is installed on your machine, you can
check the package version.
"""

import pynet
print(pynet.__version__)

#############################################################################
# Now you can run the the configuration info function to see if all the
# dependencies are installed properly.

import pynet.configure
print(pynet.configure.info())

#############################################################################
# Import a pynet dataset
# ----------------------
#
# Use a fetcher to retrieve some data and use generic interface to import and
# split this dataset: train, test and validation.
# You may need to change the 'datasetdir' parameter.

from pynet.datasets import DataManager, fetch_cifar

data = fetch_cifar(datasetdir="/neurospin/nsap/datasets/cifar")
manager = DataManager(
    input_path=data.input_path,
    labels=["label"],
    metadata_path=data.metadata_path,
    number_of_folds=10,
    batch_size=50,
    stratify_label="category",
    test_size=0.1)

#############################################################################
# We have now a test, and multiple folds with train-validation datasets that
# can be used to train our network using cross-validation.

import numpy as np
from pynet.plotting import plot_data

print("Nb folds: ", manager.number_of_folds)
dataloader = manager.get_dataloader(
    train=True,
    validation=False,
    test=False,
    fold_index=0)
print(dataloader)
for trainloader in dataloader.train:
    print("Inputs: ", trainloader.inputs.shape)
    print("Outputs: ", trainloader.outputs)
    print("Labels: ", trainloader.labels.shape)
    plot_data(trainloader.inputs, nb_samples=5)
    break

# import matplotlib.pyplot as plt
# plt.show()


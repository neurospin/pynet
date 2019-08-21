"""
pynet metastasis tumor segmentation
===================================

Credit: A Grigis

pynet is a Python package related to deep learning and its application in
MRI mediacal data analysis. It is accessible to everybody, and is reusable
in various contexts. The project is hosted on github:
https://github.com/neurospin/pynet.

In this notebook we will learn how to segment tumors in T1 MRI images.
"""

#############################################################################
# Import metastasis dataset
# -------------------------
#
# From a simple TSV file, this package provides a common interface to import,
# split and display the decribed dataset.

import pynet
from pynet.dataset import split_dataset
from pynet.dataset import LoadDataset
from pynet.transforms import ZeroPadding, Downsample
from pynet.optim import training
from pynet.models import UNet
import torch
import torch
import torch.nn as nn
import torch.optim as optim

dataset_desc = "/neurospin/radiomics_pub/workspace/metastasis_dl/data/dataset.tsv"
dataloader_kwargs = {
    "load": True
}
dataset = split_dataset(
    path=dataset_desc,
    dataloader=LoadDataset,
    batch_size=50,
    inputs=["t1"],
    outputs=["mask"],
    label="label",
    number_of_folds=1,
    transforms=[ZeroPadding(shape=(256, 256, 256)), Downsample(scale=4)],
    test_size=0.25,
    validation_size=0.1,
#    nb_samples=100,
    verbose=0,
    **dataloader_kwargs)

#############################################################################
# Check the output data size and type for one batch.

fold_index = 0
batch_index = 0
X_train = dataset["train"][fold_index][batch_index]["inputs"]
y_train = dataset["train"][fold_index][batch_index]["outputs"]
X_valid = dataset["validation"][fold_index][batch_index]["inputs"]
y_valid = dataset["validation"][fold_index][batch_index]["outputs"]
X_test = dataset["test"][batch_index]["inputs"]
y_test = dataset["test"][batch_index]["outputs"]
print(X_train.shape , y_train.shape)
print(X_valid.shape, y_valid.shape )
print(X_test.shape, y_test.shape)
print(X_train.dtype, y_train.dtype, X_train.device, y_train.device)

############################################################################
# Load a network
# --------------
#
# From the available netwoks load the 3D UNet.

net = UNet(
    num_classes=4,
    in_channels=1,
    depth=5, 
    start_filts=16,
    up_mode="upsample", 
    merge_mode="concat",
    batchnorm=True)

#############################################################################
# Optimisation
# ------------
#
# Now start the network training.

nb_epochs = 150
learning_rate = 1e-4
def my_loss(x, y):
    y = torch.sum(y, dim=1).type(torch.LongTensor)
    criterion = nn.CrossEntropyLoss()
    return criterion(x, y)
my_optimizer = optim.SGD(
    net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
test_history, train_history, valid_history = training(
    net=net,
    dataset=dataset,
    optimizer=my_optimizer,
    criterion=my_loss,
    nb_epochs=nb_epochs,
    metrics=None,
    use_cuda=False,
    outdir="/data/tmp",
    verbose=1)




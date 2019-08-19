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
# split and display the decribed dataset:

import pynet
from pynet.dataset import split_dataset
from pynet.dataset import LoadDataset
from pynet.transforms import ZeroPadding, Downsample

dataset_desc = "/neurospin/radiomics_pub/workspace/metastasis_dl/data/dataset.tsv"
dataset = split_dataset(
    path=dataset_desc,
    dataloader=LoadDataset,
    batch_size=1,
    inputs=["t1"],
    outputs=["mask"],
    label="label",
    number_of_folds=1,
    transforms=[ZeroPadding(shape=(256, 256, 256)), Downsample(scale=2)],
    verbose=0)

############################################################################
# Load a network
# --------------
#
# From the available netwoks load the 3D UNet:

import torch
from pynet.models import UNet

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
# Now start the network training

import torch
import torch.nn as nn
import torch.optim as optim
from pynet.optim import training


def my_loss(x, y):
    y = torch.sum(y, dim=1).type(torch.LongTensor)
    criterion = nn.CrossEntropyLoss()
    return criterion(x, y)


test_history, train_history, valid_history = training(
    net=net,
    dataset=dataset,
    optimizer=optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                        weight_decay=0.0005),
    criterion=my_loss,
    nb_epochs=100,
    metrics=None,
    use_cuda=False,
    outdir="/data/tmp",
    verbose=1)




if 0:
    import tqdm
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def zero_fill_256(arr):
        return zero_fill(arr, shape=(256, 256, 256))

    def downsample_128(arr):
        return downsample(arr, scale=2)


    # Check device
    use_cuda = False
    if use_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    device = torch.device("cuda" if use_cuda else "cpu")


    # Split the test/training dataset
    dataset = split_dataset(
        path="/neurospin/radiomics_pub/workspace/metastasis_dl/data/dataset.tsv",
        dataloader=LoadDataset,
        batch_size=1,
        inputs=["t1"],
        outputs=["mask"],
        label="label",
        number_of_folds=1,
        transform=[zero_fill_256, downsample_128],
        verbose=0)
    if 0:
        for b in dataset["test"]:
            print(b["inputs"].shape)
            print(b["outputs"].shape)
            print(dataset["test"].dataset.iloc[0].values)
            plot_data(b["inputs"][0, 0].numpy(), np.sum(b["outputs"][0].numpy(), axis=0))
            stop

    # Create network
    nb_classes = 4
    model = UNet(
        num_classes=nb_classes,
        in_channels=1,
        depth=5, 
        start_filts=16,
        up_mode="upsample", 
        merge_mode="concat",
        batchnorm=True)
    if 0:
        layers = get_named_layers(model)
        print(sorted(list(layers.keys())))
        stop
    if 0:
        save_network_graph(
            model=model,
            shape=[1, 1, 128, 128, 128],
            outfileroot=("/neurospin/nsap/hackathon/deeplearning-hackathon-2019/gan"
                         "/DeepLearning-SeGAN-Segmentation/unet"))
        test_model(
            model=model,
            shape=[1, 1, 128, 128, 128])

        stop
    model = model.to(device)

    # Objective function is cross-entropy
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()

    # Optimizer
    learning_rate = 0.1
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                          weight_decay=0.0005)
    nb_epochs = 2


    def train(epoch, model, dataloader):
        epoch_loss = 0
        current_loss = None
        trange = tqdm.trange(1, len(dataloader) + 1, desc="Batch")
        for iteration in trange:
            trange.set_description("Batch {0}".format(current_loss))
            trange.refresh()
            batch_data = dataloader[iteration - 1]
            inputs = batch_data["inputs"].to(device)
            y = batch_data["outputs"].to(device)
            x = model(inputs)
            y = torch.sum(y, dim=1).type(torch.LongTensor)
            loss = criterion(x, y)
            current_loss = loss.item()
            epoch_loss += current_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch[{}] finished: Loss: {:.4f}".format(epoch, epoch_loss))
        return epoch_loss


    def test(model, dataloader):
        epoch_loss = 0
        with torch.no_grad():
            trange = tqdm.trange(1, len(dataloader) + 1, desc="Test")
            for iteration in trange:
                batch_data = dataloader[iteration - 1]
                inputs = batch_data["inputs"].to(device)
                true_masks = batch_data["outputs"].to(device)
                masks_pred = model(inputs)
                y = batch_data["outputs"].to(device)
                x = model(inputs)
                y = torch.sum(y, dim=1).type(torch.LongTensor)
                loss = criterion(x, y)
                epoch_loss += loss.item()
            print("-- Test: Loss: {:.4f}".format(epoch_loss))
        return epoch_loss



    for fold in range(1, len(dataset["train"]) + 1):
        print("=" * 50)
        print("Starting fold {0}...".format(fold))
        train_accuracies = np.zeros(nb_epochs)
        test_accuracies = np.zeros(nb_epochs)
        trange = tqdm.trange(1, nb_epochs + 1, desc="Epoch")
        for epoch in trange:
            train_accuracies[epoch - 1] = train(
                epoch, model, dataset["train"][fold - 1])
            test_accuracies[epoch - 1] = test(
                model, dataset["validation"][fold - 1])
        checkpoint(
            model=model,
            epoch=epoch,
            fold=fold,
            outdir=("/neurospin/nsap/hackathon/deeplearning-hackathon-2019/gan"
                    "/DeepLearning-SeGAN-Segmentation"))
        plt.figure()
        plt.plot(np.arange(1, nb_epochs + 1), train_accuracies)
        plt.plot(np.arange(1, nb_epochs + 1), test_accuracies)
        plt.show()
        print("=" * 50)


    final_loss = test(model, dataset["test"])
    print("=" * 50)
    print("Final loss = {0}".format(final_loss))
    print("=" * 50)



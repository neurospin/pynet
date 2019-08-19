
# Imports
import torch
import torch.nn as nn
import torch.optim as optim

from model import Net
from dataset import split_dataset
from dataset import LoadDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Check device
use_cuda = False
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
device = torch.device("cuda" if use_cuda else "cpu")


# Split the test/training dataset
dataset = split_dataset(
    path="/neurospin/radiomics_pub/workspace/metastasis_dl/data/dataset.tsv",
    dataloader=LoadDataset,
    batch_size=10,
    inputs=["t1"],
    #outputs=["mask"],
    label="label",
    number_of_folds=1,
    verbose=0)
#print(dataset["test"].dataset.shape)
#for batch_data in dataset["test"]:
#    print(batch_data["inputs"].shape, batch_data["labels"].shape)
#    if batch_data["outputs"] is not None:
#        print(batch_data["outputs"].shape)


# Create network
model = Net(nb_voxels_at_layer2=44268).to(device)

# Objective function is cross-entropy
criterion = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 0.00001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
nb_epochs = 2


def train(epoch, model, dataloader):
    total = 0
    success = 0
    for iteration, batch_data in enumerate(dataloader, 1):
        inputs = batch_data["inputs"].to(device)
        labels = batch_data["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        success += (predicted == labels.data).sum()
        print("-- Epoch[{}]({}/{}): Loss: {:.4f}".format(
            epoch, iteration, len(dataloader), loss.item()))
    train_accuracy = 100.0 * success / total
    print("-- Epoch[{}]: Accuracy: {:.4f}".format(epoch, train_accuracy))

    return train_accuracy


def test(model, dataloader):
    success = 0
    total = 0
    with torch.no_grad():
        for iteration, batch_data in enumerate(dataloader, 1):
            inputs = batch_data["inputs"].to(device)
            labels = batch_data["labels"].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            success += (predicted == labels.data).sum()
            print("-- Test({}/{})".format(iteration, len(dataloader)))
        test_accuracy = 100.0 * success / total
        print("-- Test: Accuracy: {:.4f}".format(test_accuracy))
    return test_accuracy


def checkpoint(epoch, fold, model):
    model_out_path = "model_{0}_epoch_{1}.pth".format(fold, epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for fold in range(1, len(dataset["train"]) + 1):
    print("=" * 50)
    print("Starting fold {0}...".format(fold))
    train_accuracies = np.zeros(nb_epochs)
    test_accuracies = np.zeros(nb_epochs)
    for epoch in range(1, nb_epochs + 1):
        train_accuracies[epoch - 1] = train(
            epoch, model, dataset["train"][fold - 1])
        test_accuracies[epoch - 1] = test(
            model, dataset["validation"][fold - 1])
    checkpoint(epoch, fold, model)
    plt.figure()
    plt.plot(np.arange(1, nb_epochs + 1), train_accuracies)
    plt.plot(np.arange(1, nb_epochs + 1), test_accuracies)
    plt.show()
    print("=" * 50)


final_accuracy = test(model, dataset["test"])
print("=" * 50)
print("Final accuracy = {0}".format(final_accuracy))
print("=" * 50)



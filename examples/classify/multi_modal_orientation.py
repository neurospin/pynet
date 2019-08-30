"""
pynet: multi modal slice orientation prediction 
===============================================

Credit: A Grigis

This practice focuses on a meaningless toy example inspired from the MNIST
manuscript numbers classification challenge that is considered as the
'hello world' example for neural networks.

Here, we have to recognize wether a brain slice image is axial, sagittal or
coronal and comes from MRI T1w, MRI T2w or CT modality. So, there are 9
classes.

Setup
-----

First, import a few common modules, ensure matplotLib plots figures inline
and prepare a function to display the figures. Install first missing
packages with 'pip3 install --user pillow matplotlib pydot graphviz
progressbar2'.
"""

# Common imports
import numpy as np
import os
import glob

# Net tools
from pynet.models.mlp import OneLayerMLP
from pynet.dataset import split_dataset
from pynet.dataset import LoadDataset
from pynet.plotting import plot_net
from pynet.plotting import plot_data
from pynet.optim import training
from pynet.classifier import Classifier
from pynet.observers import PredictObserver

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["axes.labelsize"]  = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
def plot_image(image):
    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()
    plt.imshow(image, aspect="equal", cmap="gray", interpolation="nearest")
    plt.axis("off")


#############################################################################
# And of course we import PyTorch

import torch
torch.__version__

#############################################################################
# Read the data
# -------------
#
# We'll start by reading png files stored in the data directory. Note that the
# filename of the png image describes which view (Axial, Coronal, Sagital) and
# modality (T1w or T2w or CT) corresponds to the image.
# These png correspond to slices extracted from 3D volumes. In order to reduce
# the data variability (and simplify the slice extraction scripts), all MRI
# volumes, and respectively CT, have been registered on a reference image
# using elastix.
# Thus, the origin, spacing, orientation and number of pixels per dimensions
# of all volumes are the same and therefore will not need to be taken into
# account systematically by the learning scripts (using SimpleITK, MedPy,
# nibabel...).
# Then, slices were extracted and the window/level was adapted (using ITK).
# Finally, they were resized to 64x64 (using ImageMagick).
#
# We also define training + valid sets vs test set.
# Unfortunately, there are no clear rules to create train, valid and test sets.
# The one presented here is useful and simple when you don't need to randomize
# your inputs in order to assess size effect nor training stability using
# k-fold cross-validations.
#
# A validation step is a useful way to avoid overfitting. At each epoch, the
# neural network is evaluated on the validation set, but not trained on it.
# If the validation loss starts to grow, it means that the network is
# overfitting the training set, and that it is time to stop the training.
#
# The following cell create stratified test, train, and validation loaders.

datadir = "/neurospin/nsap/hackathon/deeplearning-hackathon-2019/tp4_data"
dataset_desc = os.path.join(datadir, "dataset.tsv")
height = 64
width  = 64
med_view = {
        "T1-A": 0,
        "T1-S": 1,
        "T1-C": 2,
        "T2-A": 3,
        "T2-S": 4,
        "T2-C": 5,
        "CT-A": 6,
        "CT-S": 7,
        "CT-C": 8
}
rev_med_view = dict((val, key) for key, val in med_view.items())
dataloader_kwargs = {
    "flatten": True,
    "squeeze_channel": True,
    "load": True
}
dataset = split_dataset(
    path=dataset_desc,
    dataloader=LoadDataset,
    inputs=["slice"],
    outputs=None,
    label="label",
    number_of_folds=1,
    number_of_batch=1,
    transforms=None,
    test_size=0.25,
    validation_size=0.1,
    verbose=0,
    nb_samples=1000,
    **dataloader_kwargs)

#############################################################################
# Check the output data size and type for one batch.

fold_index = 0
batch_index = 0
X_train = dataset["train"][fold_index][batch_index]["inputs"]
y_train = dataset["train"][fold_index][batch_index]["labels"]
X_valid = dataset["validation"][fold_index][batch_index]["inputs"]
y_valid = dataset["validation"][fold_index][batch_index]["labels"]
X_test = dataset["test"][batch_index]["inputs"]
y_test = dataset["test"][batch_index]["labels"]
print(X_train.shape , y_train.shape)
print(X_valid.shape, y_valid.shape )
print(X_test.shape, y_test.shape)
print(X_train.dtype, y_train.dtype)

#############################################################################
# Displaying some images in the train dataset.

nb_x = 5
nb_y = 5
plt.figure(figsize=(15, 15 * nb_x / nb_y), dpi=100)
for cnt in range(nb_x * nb_y):
    plt.subplot(nb_x, nb_y , cnt + 1)
    plot_image(X_train[cnt].reshape(width, height))
    plt.title(rev_med_view[y_train[cnt].item()])


#############################################################################
# Simple neural network
# ---------------------
#
# The simplest way to create, train and test a network is to use Sequential
# container.
# With a sequential container, you can quickly design a linear stack of layers
# and so, many kinds of models (LSTM, CNN, ...).
# Here we create a simple Multilayer Perceptron (MLP) for multi-class softmax
# classification.

image_size = height * width
nb_neurons = 16
# model = torch.nn.Sequential(
#     torch.nn.Linear(image_size, nb_neurons),  # first layer, fully-connected
#     torch.nn.ReLU(),                          # activation function
#    torch.nn.Dropout(p=0.1),
#     torch.nn.Linear(nb_neurons, 9),
#     torch.nn.LogSoftmax(dim=1)
# )
model = OneLayerMLP(image_size, nb_neurons, 9)
print(model)
# plot_net(model, shape=(1, image_size), static=True, outfileroot=None)

#############################################################################
#  Then we configure the parameters of the training step and train the model.

cl = Classifier(
    batch_size=50,
    optimizer_name="Adam",
    learning_rate=1e-4,
    loss_name="NLLLoss",
    metrics=["accuracy"],
    model=model,
    observers=[PredictObserver(X_valid, y_valid, "valid")])
test_history, train_history = training(
    model=cl,
    dataset=dataset,
    nb_epochs=100,
    outdir=None,
    verbose=1)
torch.save(model, os.path.join(datadir, "pytorch_mlp.pth"))

#############################################################################
# Focus on one test prediction.
# Test and evaluation on a large data set is nice for assessment, but what
# about a given result?
# The following code performs a label prediction on the image i in the test
# set.

i = 0
test_input = X_test[i].reshape(1, -1)
y_t = cl.predict_proba(test_input)
print(" ** Label probabilities: ", y_t)
print(" ** true label      : ", y_test[i] )
print(" ** predicted label : ", np.argmax(y_t))
plt.figure()
plot_image(X_test[i].reshape(width, height))
plt.title("true: " + rev_med_view[int(y_test[i])] +
          ", predict: " + rev_med_view[np.argmax(y_t)])

#############################################################################
# Watch learning curves.
# During the training, we saved the loss and the accuracy at each iteration in
# the lists losses and accuracies.
# The following lines display the corresponding curves.

_, losses = train_history["loss"]
_, accuracies = train_history["accuracy"]

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.ylabel("Training loss")
plt.xlabel("Iterations")
plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.ylabel("Training accuracy")
plt.xlabel("Iterations")

#############################################################################
# Convolutional neural network
# ----------------------------
#
# Now we will create a neural network using convolutional layers.
# A more flexible way of defining a neural network is to define a custom class.

class My_Net(torch.nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.linear = torch.nn.Linear(16 * 16 * 16, 9)

    def forward(self, X): 
        X = X.view(-1, 1, 64, 64)
        X = torch.nn.functional.relu(self.conv1(X))
        X = torch.nn.functional.relu(self.conv2(X))
        X = self.maxpool(X)
        X = self.linear(X.view(-1, 16 * 16 * 16))
        return torch.nn.functional.log_softmax(X, dim=1)

#############################################################################
# Here, we check how the input size changes through each layer.

model = My_Net()

X = torch.from_numpy(X_train[:10, :]).view(-1, 1, 64, 64)
print(X.shape)
X = torch.nn.functional.relu(model.conv1(X))
print(X.shape)
X = torch.nn.functional.relu(model.conv2(X))
print(X.shape)
X = model.maxpool(X)
print(X.shape)
X = model.linear(X.view(-1, 16 * 16 * 16))
print(X.shape)
X = torch.nn.functional.log_softmax(X, dim=1)
print(X.shape)

#############################################################################
#  Then we configure the parameters of the training step and train the model.

cl = Classifier(
    batch_size=50,
    optimizer_name="Adam",
    learning_rate=1e-5,
    loss_name="NLLLoss",
    metrics=["accuracy"],
    model=model,
    observers=[PredictObserver(X_valid, y_valid, "valid")])
test_history, train_history = training(
    model=cl,
    dataset=dataset,
    nb_epochs=100,
    outdir=None,
    verbose=1)
torch.save(model, os.path.join(datadir, "pytorch_cnn.pth"))

#############################################################################
# Focus on one test prediction.
# Test and evaluation on a large data set is nice for assessment, but what
# about a given result?
# The following code performs a label prediction on the image i in the test
# set.

i = 0
test_input = X_test[i].reshape(1, -1)
y_t = cl.predict_proba(test_input)
print(" ** Label probabilities: ", y_t)
print(" ** true label      : ", y_test[i] )
print(" ** predicted label : ", np.argmax(y_t))
plt.figure()
plot_image(X_test[i].reshape(width, height))
plt.title("true: " + rev_med_view[int(y_test[i])] +
          ", predict: " + rev_med_view[np.argmax(y_t)])

#############################################################################
# Watch learning curves.
# During the training, we saved the loss and the accuracy at each iteration in
# the lists losses and accuracies.
# The following lines display the corresponding curves.

_, losses = train_history["loss"]
_, accuracies = train_history["accuracy"]

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.ylabel("Training loss")
plt.xlabel("Iterations")
plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.ylabel("Training accuracy")
plt.xlabel("Iterations")

#############################################################################
# Compare the fully-connected network with the CNN
# ------------------------------------------------
#
# Below is a comparison in terms of trainable parameters for both models.

cnn_model = My_Net()
print("Number of parameters in the CNN: ",
      sum(p.numel() for p in cnn_model.parameters()))

image_size = height * width
nb_neurons = 16
model = OneLayerMLP(image_size, nb_neurons, 9)
print("Number of parameters in the fully connected: ",
      sum(p.numel() for p in model.parameters()))

plt.show()

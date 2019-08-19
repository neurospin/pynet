"""
pynet optim helpers overview
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
# dependencies are installed properly:

import pynet.configure
print(pynet.configure.info())


#############################################################################
# Optimisation
# ------------
#
# First load a dataset and a network

import pynet
from pynet.dataset import dummy_dataset
import torch.nn as nn
import torch.nn.functional as func

dataset = dummy_dataset(
    nb_batch=3,
    batch_size=2,
    number_of_folds=1,
    shape=(16, 16),
    verbose=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, bias=False, padding=2)
        self.conv2 = nn.Conv2d(6, 1, 1, stride=1, groups=1)
        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Max pooling over a (2, 2) window
        #x = func.max_pool2d(func.relu(self.conv1(x)), 2)
        # If the size is a square you can only specify a single number
        #x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
net = Net()

#############################################################################
# Now start the optimisation

import torch
import torch.nn as nn
import torch.optim as optim
from pynet.optim import training


def my_loss(x, y):
    # y = torch.sum(y, dim=1).type(torch.LongTensor)
    criterion = nn.MSELoss()
    return criterion(x, y)


test_history, train_history, valid_history = training(
    net=net,
    dataset=dataset,
    optimizer=optim.Adam(net.parameters(), lr=0.01),
    criterion=my_loss,
    nb_epochs=3,
    metrics={"mse": my_loss},
    use_cuda=False,
    outdir="/tmp/pynet",
    verbose=1)

#############################################################################
# You can reload the optimization history at any time and any step

from pprint import pprint
from pynet.history import History

valid_history = History.load("/tmp/pynet/history/valid_1_epoch_3.pkl")
pprint(valid_history.history)
pprint(valid_history["loss"])

#############################################################################
# You can finally display the optimization cost

from pynet.plotting import plot_data

x, y = valid_history["loss"]
plot_data(y)




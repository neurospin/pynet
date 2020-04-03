"""
pynet: hyper parameters tuning
==============================

Credit: A Grigis
Based on:
- https://github.com/autonomio/talos/blob/master/docs/Examples_PyTorch.md

In this tutorial, you will learn how to tune the hyperparameters using the
talos and the kerasplotlib modules.
"""

import talos
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_optimizer import torch_optimizer

from sklearn.metrics import f1_score

from pynet.interfaces import DeepLearningInterface
from pynet.datasets import DataManager

#############################################################################
# Data Preparation
# ----------------
#
# For this experiment, we're going to use the breast cancer dataset.

x, y = talos.templates.datasets.breast_cancer()
x = talos.utils.rescale_meanzero(x)
x_train, y_train, x_val, y_val = talos.utils.val_split(x, y, .2)
print("Train: ", x_train.shape, y_train.shape)
print("Validation: ", x_val.shape, y_val.shape)

#############################################################################
# Model Preparation
# -----------------
#
# Talos works with any pynet model, without changing the structure of the
# model in anyway, or without introducing any new syntax. The below example
# shows clearly how this works.


class BreastCancerNet(nn.Module, talos.utils.TorchHistory):
    def __init__(self, n_feature, first_neuron, second_neuron, dropout):
        super(BreastCancerNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, first_neuron)
        torch.nn.init.normal_(self.hidden.weight)
        self.hidden1 = torch.nn.Linear(first_neuron, second_neuron)
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(second_neuron, 2)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.hidden1(x))
        x = self.out(x)
        return x


def update_talos_history(signal):
        """ Callback to update talos history.

        Parameters
        ----------
        signal: SignalObject
            an object with the trained model 'object', the emitted signal
            'signal', the epoch number 'epoch' and the fold index 'fold'.
        """
        net = signal.object.model
        emitted_signal = signal.signal
        epoch = signal.epoch
        fold = signal.fold
        for key in signal.keys:
            if key in ("epoch", "fold"):
                continue
            value = getattr(signal, key)
            if value is not None:
                net.append_history(value, key)


def breast_cancer(x_train, y_train, x_val, y_val, params):
    print("Iteration parameters: ", params)

    def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)
    manager = DataManager.from_numpy(
        train_inputs=x_train, train_labels=y_train,
        batch_size=params["batch_size"], validation_inputs=x_val,
        validation_labels=y_val)
    net = BreastCancerNet(
        n_feature=x_train.shape[1], first_neuron=params["first_neuron"],
        second_neuron=params["second_neuron"], dropout=params["dropout"])
    net.apply(weights_init_uniform_rule)
    net.init_history()
    model = DeepLearningInterface(
        model=net,
        optimizer_name=params["optimizer_name"],
        learning_rate=params["learning_rate"],
        loss_name=params["loss_name"],
        metrics=["accuracy"])
    model.add_observer("after_epoch", update_talos_history)
    model.training(
        manager=manager,
        nb_epochs=params["epochs"],
        checkpointdir=None,
        fold_index=0,
        with_validation=True)
    return net, net.parameters()

#############################################################################
# Setting the Parameter Space Boundaries
# --------------------------------------
#
# In the last and final step, we're going to create the dictionary, which will
# then be passed on to Talos together with the model above. Here we have
# three different ways to input values:
# - as stepped ranges (min, max, steps)
# - as multiple values [in a list]
# - as a single value [in a list]
# For values we don't want to use, it's ok to set it as None.


params = {
    "first_neuron": [200, 100],
    "second_neuron": [30, 50],
    "dropout": [0.2, 0.3],
    "optimizer_name": ["SGD", "Adam"],
    "loss_name": ["CrossEntropyLoss"],
    "learning_rate": [1e-3, 1e-4],
    "batch_size": [20, 50, 5],
    "epochs": [10, 20]
}

#############################################################################
# Run the Hyperparameter scan
# ---------------------------
#
# Now we are ready to run the model based on the parameters and the layer
# configuration above. The exact same process would apply with any other
# model, just make sure to pass the model function name in the Scan() command
# as in the below example. To get started quickly, we're going to invoke only
# 10 rounds.

os.chdir("/tmp")
scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         params=params,
                         model=breast_cancer,
                         experiment_name="breast_cancer",
                         round_limit=10)


#############################################################################
# Access the results through the Scan object
# ------------------------------------------
#

print("accessing the results data frame")
print(scan_object.data.head())

print("accessing epoch entropy values for each round")
print(scan_object.learning_entropy)

print("access the summary details")
print(scan_object.details)

print("accessing the saved models")
print(scan_object.saved_models)

print("accessing the saved weights for models")
print(scan_object.saved_weights)

#############################################################################
# Analysing the Scan results with reporting
# -----------------------------------------
#

print("use Scan object as input")
analyze_object = talos.Analyze(scan_object)

print("access the dataframe with the results")
print(analyze_object.data)

print("get the number of rounds in the Scan")
print(analyze_object.rounds())

print("et the highest result for any metric")
print(analyze_object.high('val_accuracy'))

print("get the round with the best result")
print(analyze_object.rounds2high('val_accuracy'))

print("get the best paramaters")
print(analyze_object.best_params(
    'val_accuracy', ['accuracy', 'loss', 'val_loss']))

print("get correlation for hyperparameters against a metric")
print(analyze_object.correlate('val_loss', ['accuracy', 'loss', 'val_loss']))

print("a regression plot for two dimensions")
analyze_object.plot_regs('val_accuracy', 'val_loss')

print("line plot")
analyze_object.plot_line('val_accuracy')

print("up to two dimensional kernel density estimator")
analyze_object.plot_kde('val_accuracy')

print("a simple histogram")
analyze_object.plot_hist('val_accuracy', bins=50)

print("heatmap correlation")
analyze_object.plot_corr('val_loss', ['accuracy', 'loss', 'val_loss'])

print("a four dimensional bar grid")
analyze_object.plot_bars(
    'batch_size', 'val_accuracy', 'first_neuron', 'learning_rate')

if "CI_MODE" not in os.environ:
    import matplotlib.pyplot as plt
    plt.show()

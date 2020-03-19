"""
Practical Deep Learning for Genomic Prediction
==============================================

Credit: A Grigis

Based on:

- https://github.com/miguelperezenciso/DLpipeline

Loas the data
------------

Load some data.
You may need to change the 'datasetdir' parameter.
"""

import os
from pynet.datasets import DataManager, fetch_genomic_pred
from pynet.utils import setup_logging

setup_logging(level="info")

data = fetch_genomic_pred(
    datasetdir="/tmp/genomic_pred")
manager = DataManager(
    input_path=data.input_path,
    labels=["env0"],
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=5,
    test_size=0.2,
    continuous_labels=True)

#############################################################################
# Basic inspection

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

train_dataset = manager["train"][0]
X_train = train_dataset.inputs[train_dataset.indices]
y_train = train_dataset.labels[train_dataset.indices]
test_dataset = manager["test"]
X_test = test_dataset.inputs[test_dataset.indices]
y_test = test_dataset.labels[test_dataset.indices]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("       min max mean sd")
print("Train:", y_train.min(), y_train.max(), y_train.mean(),
      np.sqrt(y_train.var()))
print("Test:", y_test.min(), y_test.max(), y_test.mean(),
      np.sqrt(y_test.var()))
plt.figure()
plt.title("Train / test data")
plt.hist(y_train, label="Train")
plt.hist(y_test, label="Test")
plt.legend(loc="best")
X = np.concatenate((X_train, X_test))
pca = PCA(n_components=2)
p = pca.fit(X).fit_transform(X)
Ntrain = X_train.shape[0]
plt.figure()
plt.title("PCA decomposition")
plt.scatter(p[0:Ntrain,0], p[0:Ntrain,1], label="Train")
plt.scatter(p[Ntrain:, 0], p[Ntrain:, 1], label="Test", color="orange")
plt.legend(loc="best")

#############################################################################
# SNP preselection according to a simple GWAS: select N_best most
# associated SNPs or select by min_P_value.
# Optional: not used after.

from scipy import stats

pvals = []
for idx in range(X_train.shape[1]):
    b, intercept, r_value, p_value, std_err = stats.linregress(
        X_train[:, idx], y_train)
    pvals.append(-np.log10(p_value))
pvals = np.array(pvals)
plt.figure()
plt.ylabel("-log10 P-value")
plt.xlabel("SNP")
plt.plot(pvals, marker="o")
N_best = 100
snp_list = pvals.argsort()[-N_best:].squeeze().tolist()
min_P_value = 2 # P = 0.01
print(np.nonzero(pvals > min_P_value))
snp_list = np.nonzero(pvals > min_P_value)[0].squeeze().tolist()
X_train_filter = X_train[:, snp_list]
X_test_filter = X_test[:, snp_list]
print(X_train_filter.shape, y_train.shape)
print(X_test_filter.shape, y_test.shape)

#############################################################################
# Apply standard penalized methods (lasso using scikit-learn).

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

lasso = linear_model.Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
y_hat = lasso.predict(X_test)
mse = mean_squared_error(y_test, y_hat)
print("MSE in prediction =", mse)
corr = np.corrcoef(y_test, y_hat)[0, 1]
print("Corr obs vs pred =", corr)
plt.figure()
plt.title("Lasso: Observed vs Predicted Y")
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.scatter(y_test, y_hat, marker="o")

#############################################################################
# Implements a standard fully connected network (MLP) for a quantitative
# target.
# Use Mean Squared Error as loss, ie, quantitative variable, regression.
# We apply a kernel regularization on the first linear layer to punish the
# weights which are very large causing the network to overfit, after applying
# this regularization the weights will become smaller.
# We also apply an activity regularization on the first layer that tries to
# make the output smaller so as to remove overfitting. 

import collections
import torch
import torch.nn as nn
from pynet.utils import get_named_layers
from pynet.classifier import Classifier

class TwoLayersMLP(nn.Module):
    """  Simple two hidden layers percetron.
    """
    def __init__(self, data_size, nb_neurons, nb_classes, drop_rate=0.2):
        """ Initialize the instance.

        Parameters
        ----------
        data_size: int
            the number of elements in the data.
        nb_neurons: 2-uplet with int
            the number of neurons of the hidden layers.
        nb_classes: int
            the number of classes.
        drop_rate: float, default 0.2
            the dropout rate.
        """
        super(TwoLayersMLP, self).__init__()
        self.layers = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(data_size, nb_neurons[0])),
            ("activation1", nn.ReLU()),
            ("linear2", nn.Linear(nb_neurons[0], nb_neurons[1])),
            ("activation2", nn.Softplus()),
            ("drop1", nn.Dropout(drop_rate)),
            ("linear3", nn.Linear(nb_neurons[1], nb_classes))
        ]))

    def forward(self, x):
        layer1_out = self.layers[0](x)
        x = self.layers[1:](layer1_out)
        x = x.view(x.size(0))
        return x, {"layer1": layer1_out}

def linear1_l2_kernel_regularizer(signal):
    lambda2 = 0.01
    model = signal.object.model
    all_linear2_params = torch.cat([
        x.view(-1) for x in model.layers[0].parameters()])
    l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)
    return l2_regularization

def linear1_l1_activity_regularizer(signal):
    lambda1 = 0.01
    layer1_out = model = signal.layer_outputs["layer1"]
    l1_regularization = lambda1 * torch.norm(layer1_out, 1)
    return l1_regularization

if 0:
    nb_snps = X_train.shape[1]
    model = TwoLayersMLP(nb_snps, nb_neurons=[64, 32], nb_classes=1)
    print(model)
    cl = Classifier(
        optimizer_name="SGD",
        learning_rate=1e-4,
        loss_name="MSELoss",
        metrics=["pearson_correlation"],
        model=model)
    cl.add_observer("regularizer", linear1_l2_kernel_regularizer)
    cl.add_observer("regularizer", linear1_l1_activity_regularizer)
    test_history, train_history = cl.training(
        manager=manager,
        nb_epochs=100,
        checkpointdir="/tmp/genomic_pred",
        fold_index=0,
        with_validation=True)
    y_hat, X, y_true, loss, values = cl.testing(
        manager=manager,
        with_logit=False,
        predict=False)
    print(y_hat.shape, y_true.shape)
    print(y_hat)
    print(y_true)
    print("MSE in prediction =", loss)
    corr = np.corrcoef(y_true, y_hat)[0, 1]
    print("Corr obs vs pred =", corr)
    plt.figure()
    plt.title("MLP: Observed vs Predicted Y")
    plt.ylabel("Predicted")
    plt.xlabel("Observed")
    plt.scatter(y_test, y_hat, marker="o")


#############################################################################
# Implements the same fully connected network (MLP) for a quantitative
# target but in the context of multiclass target.


#############################################################################
# Implements the same probblem but witth a Convolutional Neural Network (CNN)
# for a quantitative target.


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=3, stride=3, padding=1)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(32 * 213, 64)),
            ("activation1", nn.ReLU()),
            ("linear2", nn.Linear(64, 32)),
            ("activation2", nn.Softplus()),
            ("linear3", nn.Linear(32, 1))
        ]))
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.maxpool(self.conv1(x))
        x = x.view(-1, 32 * 213)
        x = self.linear(x)
        x = x.view(x.size(0))
        return x
model = MyNet()
print(model)
cl = Classifier(
    optimizer_name="SGD",
    learning_rate=1e-4,
    loss_name="MSELoss",
    metrics=["pearson_correlation"],
    model=model)
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=10,
    checkpointdir="/tmp/genomic_pred",
    fold_index=0,
    with_validation=True)
y_hat, X, y_true, loss, values = cl.testing(
    manager=manager,
    with_logit=False,
    predict=False)
print(y_hat.shape, y_true.shape)
print(y_hat)
print(y_true)
print("MSE in prediction =", loss)
corr = np.corrcoef(y_true, y_hat)[0, 1]
print("Corr obs vs pred =", corr)
plt.figure()
plt.title("MLP: Observed vs Predicted Y")
plt.ylabel("Predicted")
plt.xlabel("Observed")
plt.scatter(y_test, y_hat, marker="o")

if "CI_MODE" not in os.environ:
    plt.show()

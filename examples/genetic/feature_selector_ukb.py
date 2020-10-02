import os
from pynet.datasets import DataManager
from pynet.feature_selection import PlinkSelector
from pynet.utils import setup_logging
import numpy as np
import pandas as pd
import progressbar
from collections import namedtuple

setup_logging(level="info")

data_path = 'neurospin/brainomics/2020_corentin_smoking'

data_file = 'nicodep_nd_aa'

Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

fam = pd.read_csv(os.path.join(data_path,'{}.fam'.format(data_file)),
    sep=' ', names=['FID', 'IID', 'father', 'mother', 'sex', 'trait'])

labels = pd.read_csv(os.path.join(data_path, 'nicodep.pheno'), sep='\t')

labels = fam[['FID', 'IID']].join(labels.set_index(['FID', 'IID']), on=['FID', 'IID'])

# data_y.drop(['fid', 'iid', 'tissue', 'mother', 'father', 'ethnicity', 'gender', 'trait', 'i', 'index'], axis=1, inplace=True
print(labels.head())
labels['smoker'] = labels['smoker'] - 1

data = Item(input_path=os.path.join(data_path, data_file),
    output_path=None, labels=None,
    metadata_path=os.path.join(data_path, 'nicodep.pheno'))

labels = ['smoker']

test_size = 0.2
n_folds = 5
n_features = 2000

feature_selector = PlinkSelector(
    kbest=n_features, data_path=data_path, data_file=data_file,
    pheno_file='nicodep.pheno', cov_file='nicodep_nd_aa.cov',
    pheno_name=labels[0], save_res_to='nicodep_res_assoc_{}_{}_{}_folds_test_{}'.format(
        data_file, labels[0], n_folds, test_size,
    ))

manager = DataManager(
    input_path=data.input_path,
    labels=labels,
    metadata_path=data.metadata_path,
    number_of_folds=n_folds,
    batch_size=32,
    test_size=test_size,
    feature_selector=feature_selector,
    continuous_labels=True)

import collections
import torch
import torch.nn as nn
from pynet.utils import get_named_layers
from pynet.interfaces import DeepLearningInterface

class LinearReg(nn.Module):
    """  Simple two hidden layers percetron.
    """
    def __init__(self, data_size=n_features):
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
        super(LinearReg, self).__init__()
        self.layers = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(data_size, 1)),
        ]))

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0))
        return x


class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=0)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2)

        self.batchnorm1 = nn.BatchNorm1d(16)
        

        out_conv1_shape = int((n_features + 2 * 0 - 1 * (3 - 1) - 1)/ 1 + 1)
        self.input_linear_features = int((out_conv1_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)

        self.dropout_linear = nn.Dropout(0.5)
        self.linear = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(16 * self.input_linear_features, 64)),
            ("activation1", nn.Softplus()),
            ("batchnorm1", nn.BatchNorm1d(64)),
            ("dropout", self.dropout_linear),
            ("linear2", nn.Linear(64, 32)),
            ("activation2", nn.Softplus()),
            ("batchnorm2", nn.BatchNorm1d(32)),
            ("dropout", self.dropout_linear),
            ("linear3", nn.Linear(32, 1)),
            ("activation3", nn.Softplus())
        ]))

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
       
        x = self.batchnorm1(nn.Softplus()(self.maxpool(self.conv1(x))))
        
        out_conv = x.view(-1, 16 * self.input_linear_features)
        x = self.linear(out_conv)
        x = x.view(x.size(0))
        return x

model = MyNet()
# model = LinearReg()


print(model)

def my_loss(x, y):
    """ nn.CrossEntropyLoss expects a torch.LongTensor containing the class
    indices without the channel dimension.
    """
    device = y.get_device()
    y = y.type(torch.FloatTensor)
    x = x.type(torch.FloatTensor)
    if device != -1:
        y = y.to(device)
    criterion = nn.MSELoss()
    return criterion(x, y)

class KernelRegularizer(object):
    """ Total Variation Loss (Smooth Term).
    For a dense flow field, we regularize it with the following loss that
    discourages discontinuity.
    k1 * FlowLoss
    FlowLoss: a gradient loss on the flow field.
    Recommend for k1 are 1.0 for ncc, or 0.01 for mse.
    """
    def __init__(self, kernel, lambda2=0.01, norm=2):
        self.kernel = kernel
        self.lambda2 = lambda2
        self.norm = norm

    def __call__(self, signal):
       def regularizer(signal):
        model = signal.object.model
        kernel = getattr(model, self.kernel)
        params = torch.cat([
            x.view(-1) for x in kernel.parameters()])
        l2_regularization = self.lambda2 * torch.norm(params, self.norm)
        return l2_regularization

cl = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=5e-4,
    loss_name="MSELoss",
    model=model,
    metrics=["pearson_correlation", "sk_r2_score"])

cl.add_observer("regularizer", KernelRegularizer('linear[0]', 0.3))
cl.add_observer("regularizer", KernelRegularizer('linear[4]', 0.3))
cl.add_observer("regularizer", KernelRegularizer('linear[8]', 0.3))
# cl.add_observer("regularizer", KernelRegularizer('layers[0]', 0.3))

train_history, valid_history = cl.training(
    manager=manager,
    nb_epochs=40,
    checkpointdir=os.path.join(data_path, "training_checkpoints"),
    fold_index=0,
    with_validation=True)

# Estimating the best number of epoch to train
best_epoch_mean = False

if best_epoch_mean:

    losses = {}

    for key, value in valid_history.history.items():
        fold, epoch = key
        if epoch in losses.keys():
            losses[epoch].append(value['loss'])
        else:
            losses[epoch] = [value['loss']]

    mean_losses = {epoch: np.mean(values) for epoch, values in losses.items()}

    best_epoch_stop = min(mean_losses, key=lambda key: mean_losses[key])

else:
    losses = {}

    for key, value in valid_history.history.items():
        fold, epoch = key
        if fold in losses.keys():
            losses[fold].append(value['loss'])
        else:
            losses[fold] = [value['loss']]

        epoch_best_loss = [np.argmin(values) for values in losses.values()]

        best_epoch_stop = int(round(np.mean(epoch_best_loss)))

print('Best number of epochs: {}'.format(best_epoch_stop))

# Preparing the training set for the model evaluation
test_dataset = manager['test']

train_indices =  np.concatenate([
    manager['train'][0].indices,
    manager['validation'][0].indices])

test_indices = test_dataset.indices

train_inputs = test_dataset.inputs[train_indices]
train_labels = test_dataset.labels[train_indices]

test_inputs = test_dataset.inputs[test_indices]
test_labels = test_dataset.labels[test_indices]

testing_manager = DataManager.from_numpy(
    train_inputs=train_inputs, train_labels=train_labels,
    test_inputs=test_inputs, test_labels=test_labels,
    continuous_labels=True, batch_size=64)

# Estimation of the model performances
n_verif = 5
final_losses = []
final_metrics = {'sk_r2_score': [], 'pearson_correlation': []}
for i in range(n_verif):
    test_history, train_history = cl.training(
        manager=testing_manager,
        nb_epochs=max(best_epoch_stop + 1, 1),
        checkpointdir=os.path.join(data_path, "training_checkpoints"),
        with_validation=False)


    y_hat, X, y_true, loss, values = cl.testing(
        manager=testing_manager,
        with_logit=False,
        predict=False)
    final_losses.append(loss)
    for name in final_metrics.keys():
        final_metrics[name].append(values[name].item())

print(final_losses)
print(final_metrics)

mean_metrics = {key: np.mean(values) for key, values in final_metrics.items()}
std_metrics = {key: np.std(values) for key, values in final_metrics.items()}

mean_loss = np.mean(final_losses)
std_loss = np.std(final_losses)

print(y_hat.shape, y_true.shape)
print(y_hat)
print(y_true)
# print(values)
# print("Loss in prediction =", loss)
print("Mean loss in prediction: {}".format(mean_loss))
print("Standard deviation of the loss in prediction: {}".format(std_loss))
print("Mean metrics in prediction: {}".format(mean_metrics))
print("Standard deviation of the metrics in prediction: {}".format(std_metrics))

import os
from pynet.datasets import DataManager
from pynet.feature_selection import PlinkSelector
from pynet.utils import setup_logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import progressbar
import shutil
from collections import namedtuple
from pandas_plink import read_plink
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge

setup_logging(level="info")

# data_path = '/neurospin/brainomics/tmp/CA263211'
data_path = '/neurospin/tmp/CA263211'

data_file = 'full_data_qc_british'

Item = namedtuple("Item", ["input_path", "output_path", "metadata_path",
                           "labels"])

# fam = pd.read_csv(os.path.join(data_path,'{}.fam'.format(data_file)),
#     sep=' ', names=['FID', 'IID', 'father', 'mother', 'sex', 'trait'])
#
# labels = pd.read_csv('/neurospin/brainomics/tmp/CA263211/ukb_rfmri_height_bmi.pheno', sep='\t')
#
# labels = fam[['FID', 'IID']].join(labels.set_index(['FID', 'IID']), on=['FID', 'IID'])
#
# _, fam, _ = read_plink(os.path.join(data_path, 'full_data_qc'))
#
# fam.sort_values('i', inplace=True)
#
# labels = pd.read_csv('/neurospin/brainomics/tmp/CA263211/ukb_rfmri_height_bmi.pheno', sep='\t')
#
# fam = fam.astype({'fid':'int64', 'iid':'int64'})
#
# labels_data = fam[['fid', 'iid']].join(labels.set_index(['FID', 'IID']), on=['fid', 'iid'])
# print(labels_data.head())

data = Item(input_path=os.path.join(data_path, data_file),
    output_path=None, labels=None,
    metadata_path=os.path.join(data_path, 'ukb_height.pheno'))
# fetch_ukb('nicodep_nd_aa', data_path, treat_nans=None)

labels = ['height']

test_size = 0.2
n_folds = 5
n_features = 10000

feature_selector = PlinkSelector(
    kbest=n_features, data_path=data_path, data_file=data_file,
    pheno_file='ukb_height.pheno', cov_file='ukb_age_sex_array_10pc.cov',
    pheno_name=labels[0], save_res_to='ukb_res_assoc_{}_{}_{}_folds_test_{}'.format(
        data_file, labels[0], n_folds, test_size,
    ))

# res_select_train = pd.read_csv(
#     os.path.join(
#         data_path,
#         'ukb_res_assoc_{}_{}_folds_test_{}'.format(
#             labels[0], n_folds, test_size,
#         ),
#         'test.assoc.linear'
#     ),
#     delim_whitespace=True)
#
# res_select_full = pd.read_csv(
#     os.path.join(
#         data_path,
#         'res_full_data_height.assoc.linear'
#     ),
#     delim_whitespace=True)
#
# res_select_full.sort_values('P', inplace=True)
# res_select_train.sort_values('P', inplace=True)
#
# for n_features in [1000, 2000, 5000, 10000]:
#     selected_full = res_select_full['SNP'].iloc[:n_features]
#     selected_train = res_select_train['SNP'].iloc[:n_features]
#
#     intersection = list(set(selected_full).intersection(set(selected_train)))
#
#     print(len(intersection))

manager = DataManager(
    input_path=data.input_path,
    labels=labels,
    metadata_path=data.metadata_path,
    number_of_folds=n_folds,
    batch_size=64,
    test_size=test_size,
    feature_selector=feature_selector,
    continuous_labels=True)

def check_indices(manager, labels_data):

    for idx, train_dataset in enumerate(manager['train']):

        train_dataset = manager["train"][idx]
        valid_dataset = manager["validation"][idx]

        y_train = train_dataset.labels[train_dataset.indices]
        y_test = valid_dataset.labels[valid_dataset.indices]

        y_t_train = labels_data.iloc[train_dataset.indices]['height'].to_numpy()
        y_t_test = labels_data.iloc[valid_dataset.indices]['height'].to_numpy()


        print(np.array_equal(y_train, y_t_train))
        print(np.array_equal(y_test, y_t_test))

# check_indices(manager, labels_data)

def correct_phenotypes(manager, file_name='full_data_qc',
    pheno_path=os.path.join(data_path, 'full_data_qc.pheno'),
    cov_path=os.path.join(data_path, 'ukb_age_10pc_sex.cov'),
    verbose=False):

    file_path = os.path.join(data_path, file_name)

    covariates = pd.read_csv(cov_path, sep=' ')
    covariates = covariates.astype({'IID': str}).set_index('IID')

    bim, fam, _ = read_plink(file_path, verbose=verbose)

    for idx, train_dataset in enumerate(manager['train']):

        train_dataset = manager["train"][idx]
        valid_dataset = manager["validation"][idx]

        y_train = train_dataset.labels[train_dataset.indices]
        y_test = valid_dataset.labels[valid_dataset.indices]

        indiv_train = fam.loc[fam['i'].isin(train_dataset.indices), ['iid','i']]
        indiv_test = fam.loc[fam['i'].isin(valid_dataset.indices), ['iid','i']]

        indiv_train = indiv_train.sort_values('i').drop(columns=['i'])
        indiv_test = indiv_test.sort_values('i').drop(columns=['i'])

        X_train = indiv_train.join(covariates, on='iid')
        X_test = indiv_test.join(covariates, on='iid')

        X_train.drop(columns=['FID', 'iid'], inplace=True)
        X_test.drop(columns=['FID', 'iid'], inplace=True)

        X_train = X_train.values
        X_test = X_test.values
        X_train = sm.add_constant(X_train)

        model = sm.OLS(y_train, X_train) # no missing='drop' expected
        results = model.fit()

        corrected_y_train = y_train - results.predict(X_train)
        corrected_y_test = y_test - results.predict(sm.add_constant(X_test))

        manager["train"][idx].labels[train_dataset.indices] = corrected_y_train
        manager["validation"][idx].labels[valid_dataset.indices] = corrected_y_test

    test_dataset = manager['test']
    train_dataset = manager['train'][0]
    valid_dataset = manager['validation'][0]

    full_train_indices = np.concatenate([train_dataset.indices, valid_dataset.indices])

    y_train = test_dataset.labels[full_train_indices]
    y_test = test_dataset.labels[test_dataset.indices]

    indiv_train = fam.loc[fam['i'].isin(full_train_indices), ['iid', 'i']]
    indiv_test = fam.loc[fam['i'].isin(test_dataset.indices), ['iid', 'i']]

    indiv_train = indiv_train.sort_values('i').drop(columns=['i'])
    indiv_test = indiv_test.sort_values('i').drop(columns=['i'])

    X_train = indiv_train.join(covariates, on='iid')
    X_test = indiv_test.join(covariates, on='iid')

    X_train.drop(columns=['FID', 'iid'], inplace=True)
    X_test.drop(columns=['FID', 'iid'], inplace=True)

    X_train = X_train.values
    X_test = X_test.values
    X_train = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train)
    results = model.fit()

    corrected_y_train = y_train - results.predict(X_train)
    corrected_y_test = y_test - results.predict(sm.add_constant(X_test))

    print(corrected_y_test)
    print(corrected_y_train)
    manager['test'].labels[full_train_indices] = corrected_y_train
    manager['test'].labels[test_dataset.indices] = corrected_y_test

def train_svm(manager, verbose=False):

    for idx, train_dataset in enumerate(manager['train']):

        train_dataset = manager["train"][idx]
        valid_dataset = manager["validation"][idx]

        y_train = train_dataset.labels[train_dataset.indices]
        y_test = valid_dataset.labels[valid_dataset.indices]

        X_train = train_dataset.inputs[train_dataset.indices]
        X_test = valid_dataset.inputs[valid_dataset.indices]

        y_train = StandardScaler().fit_transform(y_train[:, np.newaxis])
        y_test = StandardScaler().fit_transform(y_test[:, np.newaxis])

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        print(X_train.shape)
        print(X_test.shape)

        model = SVR(kernel='linear', max_iter=2000, verbose=True)
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        print('Training MSE : {}, R2 : {}'.format(mean_squared_error(y_train, pred_train), r2_score(y_train, pred_train)))
        print('Validation MSE : {}, R2 : {}'.format(mean_squared_error(y_test, pred_test), r2_score(y_test, pred_test)))

    test_dataset = manager['test']
    train_dataset = manager['train'][0]
    valid_dataset = manager['validation'][0]

    full_train_indices = np.concatenate([train_dataset.indices, valid_dataset.indices])

    y_train = test_dataset.labels[full_train_indices]
    y_test = test_dataset.labels[test_dataset.indices]

    X_train = test_dataset.inputs[full_train_indices]
    X_test = test_dataset.inputs[test_dataset.indices]

    y_train = StandardScaler().fit_transform(y_train[:, np.newaxis])
    y_test = StandardScaler().fit_transform(y_test[:, np.newaxis])

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    print(X_train.shape)
    print(X_test.shape)

    model = SVR(kernel='linear', max_iter=2000, verbose=True)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print('Training MSE : {}, R2 : {}'.format(mean_squared_error(y_train, pred_train), r2_score(y_train, pred_train)))
    print('Test MSE : {}, R2 : {}'.format(mean_squared_error(y_test, pred_test), r2_score(y_test, pred_test)))

def train_BRR(manager, verbose=False):

    for idx, train_dataset in enumerate(manager['train']):

        train_dataset = manager["train"][idx]
        valid_dataset = manager["validation"][idx]

        y_train = train_dataset.labels[train_dataset.indices]
        y_test = valid_dataset.labels[valid_dataset.indices]

        X_train = train_dataset.inputs[train_dataset.indices]
        X_test = valid_dataset.inputs[valid_dataset.indices]

        y_train = StandardScaler().fit_transform(y_train[:, np.newaxis])
        y_test = StandardScaler().fit_transform(y_test[:, np.newaxis])

        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)

        print(X_train.shape)
        print(X_test.shape)

        model = BayesianRidge()
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        print('Training MSE : {}, R2 : {}'.format(mean_squared_error(y_train, pred_train), r2_score(y_train, pred_train)))
        print('Validation MSE : {}, R2 : {}'.format(mean_squared_error(y_test, pred_test), r2_score(y_test, pred_test)))

    test_dataset = manager['test']
    train_dataset = manager['train'][0]
    valid_dataset = manager['validation'][0]

    full_train_indices = np.concatenate([train_dataset.indices, valid_dataset.indices])

    y_train = test_dataset.labels[full_train_indices]
    y_test = test_dataset.labels[test_dataset.indices]

    X_train = test_dataset.inputs[full_train_indices]
    X_test = test_dataset.inputs[test_dataset.indices]

    y_train = StandardScaler().fit_transform(y_train[:, np.newaxis])
    y_test = StandardScaler().fit_transform(y_test[:, np.newaxis])

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    print(X_train.shape)
    print(X_test.shape)

    model = BayesianRidge()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print('Training MSE : {}, R2 : {}'.format(mean_squared_error(y_train, pred_train), r2_score(y_train, pred_train)))
    print('Test MSE : {}, R2 : {}'.format(mean_squared_error(y_test, pred_test), r2_score(y_test, pred_test)))

# train_svm(manager)
# train_BRR(manager)
# correct_phenotypes(manager)
train_BRR(manager)

print(manager['test'].labels[manager['test'].indices])
print(manager['train'][0].labels[manager['train'][0].indices])
print(manager['validation'][0].labels[manager['validation'][0].indices])

import collections
import torch
import torch.nn as nn
from pynet.utils import get_named_layers
from pynet.interfaces import DeepLearningInterface


class TwoLayersMLP(nn.Module):
    """  Simple two hidden layers percetron.
    """
    def __init__(self, data_size=n_features, nb_neurons=[64, 32], nb_classes=1, drop_rate=0.2):
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
        self.nb_classes = nb_classes
        self.layers = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(data_size, nb_neurons[0])),
            ("activation1", nn.ReLU()),
            ("drop1", nn.Dropout(drop_rate)),
            ("linear2", nn.Linear(nb_neurons[0], nb_neurons[1])),
            ("activation2", nn.Softplus()),
            ("drop1", nn.Dropout(drop_rate)),
            ("linear3", nn.Linear(nb_neurons[1], nb_classes))
        ]))

    def forward(self, x):
        layer1_out = self.layers[0](x)
        x = self.layers[1:](layer1_out)
        if self.nb_classes == 1:
            x = x.view(x.size(0))
        return x, {"layer1": layer1_out}

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
        #
        #
        # self.conv2 = torch.nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=0)
        # self.batchnorm2 = nn.BatchNorm1d(16)

        # self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        # self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
        #
        # self.batchnorm1 = nn.BatchNorm1d(32)
        #
        #
        # self.conv2 = torch.nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=0)
        # self.batchnorm2 = nn.BatchNorm1d(32)
        #
        # self.conv3 = torch.nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=1)
        # self.batchnorm3 = nn.BatchNorm1d(32)
        #
        # self.conv4 = torch.nn.Conv1d(32, 16, kernel_size=10, stride=3, padding=0)
        # self.batchnorm4 = nn.BatchNorm1d(16)

        out_conv1_shape = int((n_features + 2 * 0 - 1 * (3 - 1) - 1)/ 1 + 1)
        # out_conv1_shape = int((out_conv1_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        self.input_linear_features = int((out_conv1_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)

        # out_conv2_shape = int((out_conv1_shape + 2 * 0 - 1 * (5 - 1) - 1)/ 1 + 1)
        # self.input_linear_features = int((out_conv2_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        # out_conv2_shape = int((out_conv2_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        #
        # out_conv3_shape = int((out_conv2_shape + 2 * 1 - 1 * (5 - 1) - 1)/ 1 + 1)
        # out_conv3_shape = int((out_conv3_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        #
        # out_conv4_shape = int((out_conv3_shape + 2 * 0 - 1 * (10 - 1) - 1)/ 3 + 1)
        # self.input_linear_features = int((out_conv4_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        # print(self.input_linear_features)
        self.dropout_conv = nn.Dropout(0.01)
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
        # x = self.dropout_conv(self.batchnorm1(nn.Softplus()(self.maxpool(self.conv1(x)))))
        # x = self.dropout_conv(self.batchnorm2(nn.ReLU()(self.maxpool(self.conv2(x)))))
        x = self.batchnorm1(nn.Softplus()(self.maxpool(self.conv1(x))))
        # x = self.dropout_conv(nn.Softplus()(self.maxpool(self.conv1(x))))
        # x = self.batchnorm2(nn.Softplus()(self.maxpool(self.conv2(x))))
        # x = self.batchnorm3(nn.Softplus()(self.maxpool(self.conv3(x))))
        # x = self.batchnorm4(nn.Softplus()(self.maxpool(self.conv4(x))))
        out_conv = x.view(-1, 16 * self.input_linear_features)
        x = self.linear(out_conv)
        x = x.view(x.size(0))
        return x

model = MyNet()
# model = LinearReg()

# model = TwoLayersMLP(n_features, nb_neurons=[64, 32], nb_classes=1)

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
    # momentum=0.5,
    learning_rate=5e-4,
    #loss=my_loss,
    loss_name="MSELoss",
    model=model,
    metrics=["pearson_correlation", "r2", "r2_score"])#'binary_accuracy', 'f1_score'])
    #metrics=['accuracy'])

# cl.add_observer("regularizer", KernelRegularizer('layers[0]', 0.1))
# cl.add_observer("regularizer", KernelRegularizer('layers[3]', 0.1))

cl.add_observer("regularizer", KernelRegularizer('linear[0]', 0.3))
cl.add_observer("regularizer", KernelRegularizer('linear[4]', 0.3))
cl.add_observer("regularizer", KernelRegularizer('linear[8]', 0.3))
# cl.add_observer("regularizer", KernelRegularizer('layers[0]', 0.3))

train_history, valid_history = cl.training(
    manager=manager,
    nb_epochs=40,
    # checkpointdir=os.path.join(data_path, "training_checkpoints"),
    # fold_index=0,
    with_validation=True)#,
    # early_stop=True,
    # early_stop_lag=2)

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
    # validation_inputs=test_inputs, validation_labels=test_labels,
    continuous_labels=True, batch_size=64)

n_verif = 5
final_losses = []
final_metrics = {'r2': [], 'pearson_correlation': []}
for i in range(n_verif):
    test_history, train_history = cl.training(
        manager=testing_manager,
        nb_epochs=max(best_epoch_stop + 1, 1),
        checkpointdir=os.path.join(data_path, "training_checkpoints"),
        with_validation=False)


    y_hat, X, y_true, loss, values = cl.testing(
        manager=testing_manager,
        with_logit=False,
        #logit_function='sigmoid',
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

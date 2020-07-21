import os
from pynet.datasets import DataManager, fetch_aa_nicodep
from pynet.utils import setup_logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import statsmodels.api as sm
import warnings
import progressbar

setup_logging(level="info")

data = fetch_aa_nicodep(treat_nans=None)#p_value_filter=1e-4, N_best=500)
manager = DataManager(
    input_path=data.input_path,
    labels=["smoker"],
    stratify_label="smoker",
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=5,
    test_size=0.2)

visualize_pca = False

train_dataset = manager["train"][0]
X_train = train_dataset.inputs[train_dataset.indices]
y_train = train_dataset.labels[train_dataset.indices]
test_dataset = manager["test"]
X_test = test_dataset.inputs[test_dataset.indices]
y_test = test_dataset.labels[test_dataset.indices]
valid_dataset = manager["validation"][0]
X_valid = valid_dataset.inputs[valid_dataset.indices]
y_valid = valid_dataset.labels[valid_dataset.indices]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_valid.shape, y_valid.shape)
nb_snps = X_train.shape[1]
y_train = manager["train"][0].labels[train_dataset.indices]
print(y_train.shape)

if visualize_pca:

    nan_mask_train = np.isnan(X_train.sum(axis=0))
    nan_mask_test = np.isnan(X_test.sum(axis=0))

    X_train_no_na = X_train[:, ~(nan_mask_train|nan_mask_test)]
    X_test_no_na = X_test[:, ~(nan_mask_train|nan_mask_test)]

    print(X_train_no_na.shape)
    print(X_test_no_na.shape)

    plt.figure()
    plt.title("Train / test data")
    plt.hist(y_train, label="Train")
    plt.hist(y_test, label="Test")
    plt.legend(loc="best")
    X = np.concatenate((X_train_no_na, X_test_no_na))
    pca = PCA(n_components=2)
    p = pca.fit(X).fit_transform(X)
    Ntrain = X_train.shape[0]
    plt.figure()
    plt.title("PCA decomposition")
    plt.scatter(p[0:Ntrain, 0], p[0:Ntrain, 1], label="Train")
    plt.scatter(p[Ntrain:, 0], p[Ntrain:, 1], label="Test", color="orange")
    plt.legend(loc="best")
    plt.show()


def select_features(manager, n_features, cov_file):
    covariates = pd.read_csv(cov_file)
    for idx, train_dataset in enumerate(manager['train']):


        X_train = train_dataset.inputs[train_dataset.indices]
        y_train = train_dataset.labels[train_dataset.indices]

        valid_dataset = manager["validation"][idx]
        X_valid = valid_dataset.inputs[valid_dataset.indices]
        y_valid = valid_dataset.labels[valid_dataset.indices]

        covariates_train = covariates.iloc[train_dataset.indices]
        covariates_valid = covariates.iloc[valid_dataset.indices]
        
        pbar = progressbar.ProgressBar(
                max_value=X_train.shape[1], redirect_stdout=True, prefix="Filtering snps fold {}".format(idx))

        pvals_train = []
        pvals_valid = []
        n_errors = 0
        pbar.start()
        for idx in range(X_train.shape[1]):
            pbar.update(idx+1)
            X = np.concatenate([
                X_train[:, idx, np.newaxis],
                covariates_train], axis=1)
            X = sm.add_constant(X)

            Z = np.concatenate([
                X_valid[:, idx, np.newaxis],
                covariates_valid], axis=1)
            Z = sm.add_constant(X)

            model_train = sm.Logit(y_train, X, missing='drop')
            model_valid = sm.Logit(y_valid, Z, missing='drop')

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    results_train = model_train.fit(disp=0)
                    pvals_train.append((results_train.pvalues[0]))
                except:
                    pvals_train.append(1)
                    n_errors += 1
                try:
                    results_valid = model_valid.fit(disp=0)
                    pvals_valid.append((results_valid.pvalues[0]))
                except:
                    pvals_valid.append(1)
                    n_errors += 1
        
        print('Number of errors: {}'.format(n_errors))
        pvals_train = np.array(pvals_train)
        pvals_valid = np.array(pvals_valid)

        snp_list_train = pvals_train.argsort()[:n_features].squeeze().tolist()
        manager['train'][idx].inputs = train_dataset.inputs[:, snp_list_train]

        snp_list_valid = pvals_valid.argsort()[:n_features].squeeze().tolist()
        manager['validation'][idx].inputs = valid_dataset.inputs[:, snp_list_valid]

    test_dataset = manager['test']
    X_test = test_dataset.inputs[test_dataset.indices]
    y_test = test_dataset.labels[test_dataset.indices]

    covariates_test = covariates.iloc[test_dataset.indices]
            
    pbar = progressbar.ProgressBar(
            max_value=X_test.shape[1], redirect_stdout=True, prefix="Filtering snps test ")

    pvals_test = []
    n_errors = 0
    pbar.start()
    for idx in range(X_train.shape[1]):
        pbar.update(idx+1)
        X = np.concatenate([
            X_test[:, idx, np.newaxis],
            covariates_test], axis=1)
        X = sm.add_constant(X)

        model_test = sm.Logit(y_test, X, missing='drop')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                results_test = model_test.fit(disp=0)
                pvals_test.append((results_test.pvalues[0]))
            except:
                pvals_test.append(1)
                n_errors += 1
    
    print('Number of errors: {}'.format(n_errors))
    pvals_test = np.array(pvals_test)

    snp_list = pvals_test.argsort()[:n_features].squeeze().tolist()
    manager['test'].inputs = test_dataset.inputs[:, snp_list]

print('Start feature selection')

print(manager['test'].inputs.shape)
select_features(manager, 500, '/leQ')
print(manager['test'].inputs.shape)

import collections
import torch
import torch.nn as nn
from pynet.utils import get_named_layers
from pynet.interfaces import DeepLearningInterface


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
        self.nb_classes = nb_classes
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
        if self.nb_classes == 1:
            x = nn.Sigmoid()(x.squeeze())
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


nb_snps = X_train.shape[1]
model = TwoLayersMLP(nb_snps, nb_neurons=[64, 32], nb_classes=1)
print(model)
# cl = DeepLearningInterface(
#     optimizer_name="SGD",
#     learning_rate=5e-4,
#     loss_name="MSELoss",
#     metrics=["pearson_correlation"],
#     model=model)
# cl.add_observer("regularizer", linear1_l2_kernel_regularizer)
# cl.add_observer("regularizer", linear1_l1_activity_regularizer)
# test_history, train_history = cl.training(
#     manager=manager,
#     nb_epochs=(100 if "CI_MODE" not in os.environ else 10),
#     checkpointdir="/tmp/genomic_pred",
#     fold_index=0,
#     with_validation=True)
# y_hat, X, y_true, loss, values = cl.testing(
#     manager=manager,
#     with_logit=False,
#     predict=False)
# print(y_hat.shape, y_true.shape)
# print(y_hat)
# print(y_true)
# print("MSE in prediction =", loss)
# corr = np.corrcoef(y_true, y_hat)[0, 1]
# print("Corr obs vs pred =", corr)
# plt.figure()
# plt.title("MLP: Observed vs Predicted Y")
# plt.ylabel("Predicted")
# plt.xlabel("Observed")
# plt.scatter(y_test, y_hat, marker="o")

def my_loss(x, y):
    """ nn.CrossEntropyLoss expects a torch.LongTensor containing the class
    indices without the channel dimension.
    """
    device = y.get_device()
    if y.ndim > 1:
        y = torch.argmax(y, dim=1).type(torch.LongTensor)
        criterion = nn.CrossEntropyLoss()
    else:
        y = y.type(torch.FloatTensor)
        if device != -1:
            y = y.to(device)
        criterion = nn.BCELoss()#WithLogitsLoss()
    return criterion(x, y)


cl = DeepLearningInterface(
    optimizer_name="Adam",
    learning_rate=3e-5,
    loss=my_loss,
    #loss_name="BCELoss",
    model=model,
    metrics=['binary_precision', 'binary_recall'])#, 'f1_score'])
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=20,
    checkpointdir="/neurospin/brainomics/2020_corentin_smoking/training_checkpoints",
    fold_index=0,
    with_validation=True)
y_hat, X, y_true, loss, values = cl.testing(
    manager=manager,
    with_logit=False,
    #logit_function='sigmoid',
    predict=False)
print(y_hat.shape, y_true.shape)
print(y_hat)
print(y_true)
print("Crossentropy in prediction =", loss)
# heat = np.zeros([3, 3])
# for i in range(3):
#     klass = np.nonzero(y_true[:, i] > 0)
#     for j in range(3):
#         heat[i, j] = np.mean(y_hat[klass, j])
# print("Probabilities matrix", heat)
# plt.figure()
# plot = plt.imshow(heat, cmap="Blues")
# plt.ylabel("Predicted class")
# plt.xlabel("Observed class")
# plt.show()

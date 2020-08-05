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
import pickle
import shutil
from pandas_plink import read_plink
import subprocess

setup_logging(level="info")

data_path = '/neurospin/brainomics/2020_corentin_smoking/'
manager_path = os.path.join(data_path, 'data_manager.pkl')
if not os.path.exists(manager_path):

    data = fetch_aa_nicodep(data_path, treat_nans=None)

    labels = ['smoker']

    manager = DataManager(
        input_path=data.input_path,
        labels=labels,
        stratify_label="smoker",
        metadata_path=data.metadata_path,
        number_of_folds=4,
        batch_size=16,
        test_size=0.2)

    visualize_pca = False

    if visualize_pca:

        train_dataset = manager["train"][0]
        X_train = train_dataset.inputs[train_dataset.indices]
        y_train = train_dataset.labels[train_dataset.indices]
        test_dataset = manager["test"]
        X_test = test_dataset.inputs[test_dataset.indices]
        y_test = test_dataset.labels[test_dataset.indices]

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


    def select_features(manager, n_features, data_name, label=None,
        use_plink=True, plink_path=os.path.join(data_path, 'plink'),
        plink_maf=0.01, plink_method='logistic',
        pheno_path=os.path.join(data_path, 'nicodep.pheno')):

        if not use_plink:
            covariates = pd.read_csv(cov_file, sep=' ')
            covariates.drop(['FID', 'IID'], axis=1, inplace=True)
        for idx, train_dataset in enumerate(manager['train']):

            X_train = train_dataset.inputs[train_dataset.indices]
            y_train = train_dataset.labels[train_dataset.indices]

            valid_dataset = manager["validation"][idx]

            if use_plink:
                if not os.path.isdir(os.path.join(data_path, 'tmp')):
                    os.mkdir(os.path.join(data_path, 'tmp'))

                bim, fam, _ = read_plink(os.path.join(data_path, 'nicodep_nd_aa'),
                    verbose=False)

                indiv_to_keep = fam[['fid', 'iid']].iloc[train_dataset.indices]
                indiv_to_keep.to_csv(os.path.join(data_path, 'tmp', 'indivs.txt'),
                    header=False, index=False, sep=' ')

                mask_to_remove = np.isnan(train_dataset.inputs.sum(axis=0))
                snp_to_remove = np.arange(train_dataset.inputs.shape[1])[mask_to_remove]

                snp_to_remove = bim.loc[bim['i'].isin(snp_to_remove), 'snp']
                snp_to_remove.to_csv(os.path.join(data_path, 'tmp', 'snps.txt'),
                    header=False, index=False, sep='\n')


                file_path = os.path.join(data_path, data_name)

                print('Feature selection fold {}'.format(idx))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    if not label:
                        subprocess.run([
                            os.path.join(plink_path, 'plink'),
                            '--bfile', file_path,
                            '--maf', str(plink_maf),
                            '--geno', str(0),
                            '--keep', os.path.join(data_path, 'tmp', 'indivs.txt'),
                            '--exclude', os.path.join(data_path, 'tmp', 'snps.txt'),
                            '--{}'.format(plink_method), '--covar', file_path + '.cov',
                            '--out', os.path.join(data_path, 'tmp', 'res')],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
                    else:
                        subprocess.run([
                            os.path.join(plink_path, 'plink'),
                            '--bfile', file_path,
                            '--maf', str(plink_maf),
                            '--geno', str(0),
                            '--keep', os.path.join(data_path, 'tmp', 'indivs.txt'),
                            '--exclude', os.path.join(data_path, 'tmp', 'snps.txt'),
                            '--{}'.format(plink_method), '--covar', file_path + '.cov',
                            '--pheno', pheno_path, '--pheno-name', label,
                            '--out', os.path.join(data_path, 'tmp', 'res')],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

                res = pd.read_csv(os.path.join(data_path, 'tmp', 'res.assoc.{}'.format(plink_method)), delim_whitespace=True)
                res = res[res['TEST'] == 'ADD']

                ordered_best_res = res.sort_values('P')
                best_snp_rs = res['SNP'].iloc[:n_features]
                snp_list = bim.loc[bim['snp'].isin(best_snp_rs), ['chrom', 'pos', 'i']]
                snp_list = snp_list.sort_values(['chrom', 'pos'])['i'].tolist()
            else:
                covariates_train = covariates.iloc[train_dataset.indices]

                pbar = progressbar.ProgressBar(
                        max_value=X_train.shape[1], redirect_stdout=True, prefix="Filtering snps fold {}".format(idx))

                pvals = []
                n_errors = 0
                pbar.start()
                for i in range(X_train.shape[1]):
                    pbar.update(i+1)
                    X = np.concatenate([
                        X_train[:, i, np.newaxis],
                        covariates_train.values], axis=1)

                    X = sm.add_constant(X)

                    model = sm.Logit(y_train, X, missing='drop')

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        try:
                            results = model.fit(disp=0)
                            pvals.append((results.pvalues[0]))
                        except:
                            pvals.append(1)
                            n_errors += 1

                pbar.finish()
                print('Number of errors: {}'.format(n_errors))
                pvals = np.array(pvals)

                snp_list = np.sort(pvals.argsort()[:n_features].squeeze()).tolist()

            manager['train'][idx].inputs = train_dataset.inputs[:, snp_list]

            manager['validation'][idx].inputs = valid_dataset.inputs[:, snp_list]



        test_dataset = manager['test']

        if use_plink:
            if not os.path.isdir(os.path.join(data_path, 'tmp')):
                os.mkdir(os.path.join(data_path, 'tmp'))

            bim, fam, _ = read_plink(os.path.join(data_path, 'nicodep_nd_aa'),
                verbose=False)

            to_remove = fam[['fid', 'iid']].iloc[test_dataset.indices]
            to_remove.to_csv(os.path.join(data_path, 'tmp', 'indivs.txt'),
                header=False, index=False, sep=' ')

            mask_to_remove = np.isnan(test_dataset.inputs.sum(axis=0))
            snp_to_remove = np.arange(test_dataset.inputs.shape[1])[mask_to_remove]

            snp_to_remove = bim.loc[bim['i'].isin(snp_to_remove), 'snp']
            snp_to_remove.to_csv(os.path.join(data_path, 'tmp', 'snps.txt'),
                header=False, index=False, sep='\n')

            file_path = os.path.join(data_path, data_name)

            print('Feature selection for testing'.format(idx))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if not label:
                    subprocess.run([
                        os.path.join(plink_path, 'plink'),
                        '--bfile', file_path,
                        '--maf', str(plink_maf),
                        '--geno', str(0),
                        '--remove', os.path.join(data_path, 'tmp', 'indivs.txt'),
                        '--exclude', os.path.join(data_path, 'tmp', 'snps.txt'),
                        '--{}'.format(plink_method), '--covar', file_path + '.cov',
                        '--out', os.path.join(data_path, 'tmp', 'res')],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                else:
                    subprocess.run([
                        os.path.join(plink_path, 'plink'),
                        '--bfile', file_path,
                        '--maf', str(plink_maf),
                        '--geno', str(0),
                        '--keep', os.path.join(data_path, 'tmp', 'indivs.txt'),
                        '--exclude', os.path.join(data_path, 'tmp', 'snps.txt'),
                        '--{}'.format(plink_method), '--covar', file_path + '.cov',
                        '--pheno', pheno_path, '--pheno-name', label,
                        '--out', os.path.join(data_path, 'tmp', 'res')],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

            res = pd.read_csv(os.path.join(data_path, 'tmp', 'res.assoc.{}'.format(plink_method)), delim_whitespace=True)
            res = res[res['TEST'] == 'ADD']

            ordered_best_res = res.sort_values('P')
            best_snp_rs = res['SNP'].iloc[:n_features]
            snp_list = bim.loc[bim['snp'].isin(best_snp_rs), ['chrom', 'pos', 'i']]
            snp_list = snp_list.sort_values(['chrom', 'pos'])['i'].tolist()

        else:
            train_dataset = manager['train'][0]
            valid_dataset = manager['validation'][0]

            full_train_indices = np.concatenate([train_dataset.indices, valid_dataset.indices])

            covariates_full_train = covariates.iloc[full_train_indices]
            full_X_train = test_dataset.inputs[full_train_indices]
            full_y_train = test_dataset.labels[full_train_indices]

            pbar = progressbar.ProgressBar(
                    max_value=full_X_train.shape[1], redirect_stdout=True, prefix="Filtering snps test ")

            pvals = []
            n_errors = 0
            pbar.start()
            for idx in range(full_X_train.shape[1]):
                pbar.update(idx+1)
                X = np.concatenate([
                    full_X_train[:, idx, np.newaxis],
                    covariates_full_train.values], axis=1)
                X = sm.add_constant(X)

                model = sm.Logit(full_y_train, X, missing='drop')

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    try:
                        results = model.fit(disp=0)
                        pvals.append((results.pvalues[0]))
                    except:
                        pvals.append(1)
                        n_errors += 1
            pbar.finish()
            print('Number of errors: {}'.format(n_errors))
            pvals = np.array(pvals)

            snp_list = np.sort(pvals.argsort()[:n_features].squeeze()).tolist()
        manager['test'].inputs = test_dataset.inputs[:, snp_list]

        if use_plink:
            shutil.rmtree(os.path.join(data_path, 'tmp'), ignore_errors=True)

    print('Start feature selection')

    select_features(manager, 5000, 'nicodep_nd_aa')#, plink_method='linear', label='ftnd')
    # with open(manager_path, 'wb') as output:
    #     pickle.dump(manager, output, pickle.HIGHEST_PROTOCOL)
else:

    print('Loading data manager')
    with open(manager_path, 'rb') as input:
        manager = pickle.load(input)


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
        x.view(-1) for x in model.linear[0].parameters()])
    l2_regularization = lambda2 * torch.norm(all_linear2_params, 2)
    return l2_regularization


def linear1_l1_activity_regularizer(signal):
    lambda1 = 0.01
    layer1_out = signal.layer_outputs["layer1"]
    l1_regularization = lambda1 * torch.norm(layer1_out, 1)
    return l1_regularization


nb_snps = manager['train'][0].inputs.shape[1]
model = TwoLayersMLP(nb_snps, nb_neurons=[128, 32], nb_classes=1)

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=8, stride=8, padding=1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2)

        self.batchnorm1 = nn.BatchNorm1d(64)


        self.conv2 = torch.nn.Conv1d(64, 32, kernel_size=20, stride=8, padding=0)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=2)
        self.batchnorm2 = nn.BatchNorm1d(32)

        out_conv1_shape = int((nb_snps + 2 * 1 - 1 * (8 - 1) - 1)/ 8 + 1)
        out_conv1_shape = int((out_conv1_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)

        out_conv2_shape = int((out_conv1_shape + 2 * 0 - 1 * (20 - 1) - 1)/ 8 + 1)
        self.input_linear_features = int((out_conv2_shape + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        self.dropout = nn.Dropout(0.6)
        self.linear = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(32 * self.input_linear_features, 64)),
            ("activation1", nn.ReLU()),
            ("batchnorm1", nn.BatchNorm1d(64)),
            ("dropout", self.dropout),
            ("linear2", nn.Linear(64, 32)),
            ("activation2", nn.Softplus()),
            ("batchnorm2", nn.BatchNorm1d(32)),
            ("dropout", self.dropout),
            ("linear3", nn.Linear(32, 1))
        ]))

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.batchnorm1(nn.Relu()(self.maxpool1(self.conv1(x))))
        x = self.batchnorm2(nn.Relu()(self.maxpool2(self.conv2(x))))
        out_conv = x.view(-1, 32 * self.input_linear_features)
        x = self.linear(out_conv)
        x = x.view(x.size(0))
        x = nn.Sigmoid()(x)
        return x, {"layer1": out_conv}


model = MyNet()
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
        #x = x.type(torch.FloatTensor)
        if device != -1:
            y = y.to(device)
        criterion = nn.BCELoss()#WithLogitsLoss()
    return criterion(x, y)


cl = DeepLearningInterface(
    optimizer_name="Adam",
    #momentum=0.8,
    learning_rate=5e-5,
    loss=my_loss,
    #loss_name="MSELoss",
    model=model,
    metrics=['binary_accuracy', 'binary_precision', 'binary_recall', 'f1_score'])
    #metrics=['accuracy'])

# cl.add_observer("regularizer", linear1_l2_kernel_regularizer)
#cl.add_observer("regularizer", linear1_l1_activity_regularizer)
test_history, train_history = cl.training(
    manager=manager,
    nb_epochs=40,
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

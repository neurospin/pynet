# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Deep Clustering for Unsupervised Learning of Visual Features.
"""

# Imports
import logging
import datetime
import numpy as np
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.decomposition import PCA

# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("classifier", ))
class DeepCluster(nn.Module):
    """ Deep Clustering for Unsupervised Learning of Visual Features.
    """
    def __init__(self, network, clustering, data_loader, n_batchs, pca_dim=256,
                 assignment_logfile=None, use_cuda=False):
        """ Init class.

        Parameters
        ----------
        network: @callable
            the network used to compute the features.
        clustering: @callable
            the clustering algorithm.
        data_loader: DataLoader
            the train data loader.
        n_batchs: int
            the number of batchs used to computes network features.
        pca_dim: int, default 256
            the dimension of input clustering features.
        assignment_logfile: str, default None
            save the cluster assignements at each epoch.
        use_cuda: bool, default False
            wether to use GPU or CPU.
        """
        super(DeepCluster, self).__init__()
        self.network = network
        self.clustering = clustering
        self.data_loader = data_loader
        self.n_batchs = n_batchs
        self.pca_dim = pca_dim
        self.assignment_logfile = assignment_logfile
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._write("DeepCluster: " + datetime.datetime.now().isoformat())
        if len(self.data_loader.dataset.input_transforms) != 0:
            raise ValueError(
                "Data transformation/augmentation no yet supported.")

    def update_pseudo_labels(self):
        """ Update the classification labels.
        """
        logger.debug("Update pseudo labels...")

        # Get the features for the whole dataset
        features = self.compute_features()

        # Cluster the features
        labels = self.cluster(features)

        # Assign pseudo-labels
        self.data_loader.dataset.labels = labels

        return labels

    def forward(self, x):
        """ Forward method.

        Parameters
        ----------
        x: Tensor (batch, channels, *dims)
            the input data.
        """
        return self.network(x)

    def compute_features(self):
        """ Compute the network features.

        Returns
        -------
        features: array (N, ndim)
            network features.
        """
        logger.debug("compute features:")
        # Todo: apply dataloader indices
        data = self.data_loader.dataset.inputs.astype(np.float32)
        logger.debug("- data: {0}".format(data.shape))
        batchs = np.array_split(data, self.n_batchs)

        self.network.eval()
        with torch.no_grad():
            y = []
            features = []
            for iteration, arr in enumerate(batchs):
                logger.debug("- iteration {0}/{1}: {2}".format(
                    iteration, self.n_batchs, arr.shape))
                inputs = torch.from_numpy(arr).to(self.device)
                output_items = self.network(inputs)
                if (not isinstance(output_items, tuple) and
                        not isinstance(output_items, list)):
                    raise ValueError(
                        "The network needs to return two values: the network "
                        "prediction and a dictionary with the 'features'.")
                if (len(output_items) != 2 or
                        not isinstance(output_items[1], dict) or
                        "features" not in output_items[1]):
                    raise ValueError(
                        "The network needs to return two values: the network "
                        "prediction and a dictionary with the 'features'.")
                features.append(output_items[1]["features"].data.cpu().numpy())
                logger.debug("- features: {0}".format(features[-1].shape))
            features = np.concatenate(features, axis=0)
            logger.debug("- features: {0}".format(features.shape))

        return features

    def preprocess_features(self, features):
        """ Preprocess the network features.

        Parameters
        ----------
        features: array (N, ndim)
            network features to preprocess.

        Returns
        -------
        features: array (N, pca_dim)
            PCA-reduced, whitened and L2-normalized features.
        """
        # Apply PCA-whitening
        features = features.astype("float32")
        logger.debug("- features: {0}".format(features.shape))
        pca = PCA(n_components=self.pca_dim, whiten=True)
        pca.fit(features)
        features = pca.transform(features)
        logger.debug("- PCA reduced features: {0}".format(features.shape))

        # L2 normalization
        row_sums = np.linalg.norm(features, axis=1)
        features = features / row_sums[:, np.newaxis]

        return features

    def cluster(self, features):
        """ Performs the clustering.

        Parameters
        ----------
        features: array (N, ndim)
            network features to preprocess.

        Returns
        -------
        labels: array (N, )
            the predicted class assignments.
        """
        # PCA-reducing, whitening and L2-normalization
        logger.debug("preprocess features:")
        xb = self.preprocess_features(features)

        # Cluster the data
        logger.debug("cluster data:")
        if hasattr(self.clustering, "cluster_centers_"):
            self.clustering.init = self.clustering.cluster_centers_
        self.clustering.fit(xb)
        labels = self.clustering.predict(xb)
        logger.debug("- labels: {0}".format(labels.shape))

        # Save assignements
        self._write(",".join([str(e) for e in labels]))

        return labels

    def _write(self, value):
        """ Write in log.

        Parameters
        ----------
        value: str
            the value to be written.
        """
        if self.assignment_logfile is not None:
            with open(self.assignment_logfile, "at") as open_file:
                open_file.write(value)
                open_file.write("\n")


def update_pseudo_labels(signal):
    """ Callback to update the classifier pseudo labels.

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
    labels = net.update_pseudo_labels()
    if hasattr(signal.object, "board"):
        board = signal.object.board
        board.viewer.bar(
            labels,
            win="pseudo_labels",
            opts={
                "title": "epoch {0}".format(epoch),
                "caption": "pseudo_labels"})

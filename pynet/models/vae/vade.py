# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Variational Deep Embedding (VaDE).
"""

# Imports
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from pynet.interfaces import DeepLearningDecorator
from pynet.utils import Networks, init_weight


# Global parameters
logger = logging.getLogger("pynet")


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae", "classifier"))
class VaDENet(torch.nn.Module):
    """ Variational Deep Embedding (VaDE) Network.

    Variational Deep Embedding: An Unsupervised and Generative Approach
    to Clustering, Zhuxi Jiang, arXiv 2017.
    """
    def __init__(self, n_classes, input_dim, latent_dim,
                 hidden_dims=[500, 500, 2000],  binary=True):
        """ Init classs.

        Parameters
        -----------
        n_classes: int
            the number of clusters.
        input_dim: int
            the dimension of observed data.
        latent_dim: int
            the dimension of latent space.
        hidden_dims: list of int, default [500, 500, 2000]
            the network hidden dimensions.
        binary: bool, default False
            apply sigmoid to get binary output.
        data: Tensor (N, M), default None
            data that can be used during display callbacks.
        labels: list of int (N, ), default None
            labels that can be used during display callbacks.
        """
        super(VaDENet, self).__init__()

        # Parameters
        self.n_classes = n_classes
        self.binary = binary
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # GMM parameters
        self.theta_p = nn.Parameter(
            torch.ones(n_classes) / n_classes, requires_grad=True)  # pi
        self.u_p = nn.Parameter(
            torch.zeros(latent_dim, n_classes), requires_grad=True)
        self.lambda_p = nn.Parameter(
            torch.ones(latent_dim, n_classes), requires_grad=True)

        # Dense VAE
        self.encoder = self.init_layers(
            [input_dim] + hidden_dims, activation="relu", dropout=0)
        self.encoder_mu = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = self.init_layers(
            [latent_dim] + hidden_dims[::-1], activation="relu", dropout=0)
        if binary:
            self.decoder_mean = nn.Sequential(
                nn.Linear(hidden_dims[0], input_dim),
                nn.Sigmoid())
        else:
            self.decoder_mean = nn.Linear(hidden_dims[0], input_dim)

    def init_layers(self, layer_dims, activation="relu", dropout=0.5):
        layers = []
        for idx in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[idx - 1], layer_dims[idx]))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError("Unsupported activation.")
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        pretrained_dict = torch.load(
            path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {key: val for key, val in pretrained_dict.items()
                           if key in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def init_gmm(self, dataloader, device):
        self.eval()
        data = []
        for iteration, dataitem in enumerate(dataloader):
            inputs = dataitem.inputs.to(device)
            inputs = Variable(inputs)
            _, extra = self.forward(inputs)
            z = extra["z"]
            data.append(z.data.cpu().numpy())
        data = np.concatenate(data)

        gmm = GaussianMixture(
            n_components=self.n_classes, covariance_type="diag")
        pred = gmm.fit_predict(data)

        # model.theta_p.data.copy_(torch.from_numpy(gmm.weights_).float())
        model.u_p.data.copy_(torch.from_numpy(gmm.means_.T).float())
        model.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T).float())

    @property
    def weights(self):
        return torch.softmax(self.theta_p, dim=0)

    def encode(self, x):
        h = self.encoder(x)
        z_mu = self.encoder_mu(h)
        z_logvar = self.encoder_logvar(h)
        return z_mu, z_logvar

    def decode(self, z):
        return self.decoder_mean(self.decoder(z))

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        recon_x = self.decode(z)
        return recon_x, {"z": z, "z_mu": z_mu, "z_logvar": z_logvar,
                         "model": self}

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z

    def get_gamma(self, z, z_mu, z_logvar):
        # NxDxK
        z_t = z.unsqueeze(dim=2).expand(
            z.size()[0], z.size()[1], self.n_classes)
        u_t = self.u_p.unsqueeze(dim=0).expand(
            z.size()[0], self.u_p.size()[0], self.u_p.size()[1])
        lambda_t = self.lambda_p.unsqueeze(dim=0).expand(
            z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])

        # NxK
        theta_t = self.theta_p.unsqueeze(dim=0).expand(
            z.size()[0], self.n_classes)

        # NxK
        p_c_z = (
            torch.exp(
                torch.log(theta_t) -
                torch.sum(0.5 * torch.log(2 * math.pi * lambda_t)
                          + (z_t - u_t)**2 / (2 * lambda_t), dim=1))
            + 1e-10)
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma

    def predict(self, x):
        with torch.no_grad():
            z_mu, z_logvar = self.encode(x)
            z = self.reparameterize(z_mu, z_logvar)
            gamma = self.get_gamma(z, z_mu, z_logvar)
            gamma = gamma.detach().cpu().numpy()
        pred = np.argmax(gamma, axis=1)
        return pred


@Networks.register
@DeepLearningDecorator(family=("encoder", "vae"))
class VaDEPreTrainNet(torch.nn.Module):
    """ Auto-Encoder for pretraining VaDE.

    Variational Deep Embedding: An Unsupervised and Generative Approach
    to Clustering, Zhuxi Jiang, arXiv 2017.
    """
    def __init__(self, model):
        """ Init class.

        Parameters
        ----------
        model: VaDE
            the model.
        """
        super(VaDEPreTrainNet, self).__init__()
        self.model = model

    def encode(self, x):
        return self.model.encoder_mu(self.model.encoder(x))

    def decode(self, z):
        return self.model.decode(z)

    def forward(self, x):
        z_mu = self.encode(x)
        recon_x = self.decode(z_mu)
        return recon_x

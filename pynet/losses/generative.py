# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides generative losses.

Code: https://github.com/YannDubs/disentangling-vae
"""

# Imports
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from torch.nn import functional as func
from torch.distributions import Bernoulli, Normal, Laplace, kl_divergence
from pynet.utils import Losses


def get_vae_loss(loss_name, **kwargs):
    """ Return the correct VAE loss function given the input arguments.

    The parameters for each loss:

    - vae: -
    - betah: beta
    - betab: C_init, C_fin, gamma
    - factor: device, gamma, latent_dim, lr_disc
    - btcvae: dataset_size, alpha, beta, gamma
    - sparse: beta

    Parameters
    ----------
    loss_name: str
        the name of the loss.
    kwargs: dict
        the loss kwargs.

    Returns
    -------
    loss: @callable
        the loss function.
    """
    common_kwargs = dict(steps_anneal=kwargs["steps_anneal"],
                         use_mse=kwargs["use_mse"])

    if loss_name == "betah":
        loss = BetaHLoss(beta=kwargs["beta"], **common_kwargs)
    elif loss_name == "vae":
        loss = BetaHLoss(beta=1, **common_kwargs)
    elif loss_name == "betab":
        loss = BetaBLoss(C_init=kwargs["C_init"], C_fin=kwargs["C_fin"],
                         gamma=kwargs["gamma"], **common_kwargs)
    elif loss_name == "factor":
        loss = FactorKLoss(gamma=kwargs["gamma"],
                           disc_kwargs=dict(latent_dim=kwargs["latent_dim"]),
                           optim_kwargs=dict(lr=kwargs["lr_disc"],
                                             betas=(0.5, 0.9)),
                           **common_kwargs)
    elif loss_name == "btcvae":
        loss = BtcvaeLoss(dataset_size=kwargs["dataset_size"],
                          alpha=kwargs["alpha"], beta=kwargs["beta"],
                          gamma=kwargs["gamma"], is_mss=kwargs["is_mss"],
                          **common_kwargs)
    elif loss_name == "sparse":
        loss = SparseLoss(beta=kwargs["beta"], **common_kwargs)
    else:
        raise ValueError("Uknown loss: {}".format(loss_name))

    return loss


class BaseLoss(object):
    """ Base class for losses.
    """
    def __init__(self, steps_anneal=0, use_mse=False):
        """ Init class.

        Parameters
        ----------
        steps_anneal: int, default 0
            number of annealing steps where gradually adding the
            regularisation.
        use_mse: bool, default False
            if set use MSE for the reconstruction loss rather than Log
            Likelihood.
        """
        self.n_train_steps = 0
        self.layer_outputs = None
        self.steps_anneal = steps_anneal
        self.use_mse = use_mse
        self.cache = {}

    def get_params(self):
        """ Get forward layers outputs.

        Returns
        -------
        q: torch.distributions
            probabilistic encoder (or estimated posterior probability
            function).
        z: torch.Tensor
            the compressed code learned in the bottleneck layer.
        model: nn.Module
            the network.
        """
        if self.layer_outputs is None:
            raise ValueError("The model needs to return the latent space "
                             "distribution parameters q and sampling z as "
                             "well as the model itself.")
        z = self.layer_outputs["z"]
        q = self.layer_outputs["q"]
        model = self.layer_outputs["model"]
        return q, z, model

    def reconstruction_loss(self, p, data):
        """ Calculates the per image reconstruction loss for a batch of data
        (i.e. negative log likelihood).

        The distribution of the likelihood on the each pixel implicitely
        defines the loss. Bernoulli corresponds to a binary cross entropy.
        Gaussian distribution corresponds to MSE, and is sometimes
        used, but hard to train because it ends up focusing only a few
        pixels that are very wrong. Laplace distribution corresponds to
        L1 solves partially the issue of MSE.

        Parameters
        ----------
        p: torch.distributions
            probabilistic decoder (or likelihood of generating true data
            sample given the latent code).
        data: torch.Tensor
            reference data.

        Returns
        -------
        loss: torch.Tensor
            per image cross entropy (i.e. normalized per batch but not pixel
            and channel).
        """
        if isinstance(p, Bernoulli):
            loss = func.binary_cross_entropy(p.probs, data, reduction="sum")
        elif isinstance(p, Normal):
            if self.use_mse:
                loss = func.mse_loss(p.loc, data, reduction="sum")
            else:
                loss = self.compute_ll(p, data)
        elif isinstance(p, Laplace):
            loss = func.l1_loss(p.loc, data, reduction="sum")
            # empirical value to give similar values than bernoulli => use
            # same hyperparam
            loss = loss * 3
            loss = loss * (loss != 0)  # masking to avoid nan
        else:
            raise ValueError("Unkown distribution: {}".format(distribution))

        batch_size = len(data)
        loss = loss / batch_size

        if "iteration" not in self.cache:
            self.cache["iteration"] = 0
        self.cache["iteration"] += len(data)
        self.cache.setdefault("ll", []).append(loss.detach().cpu().numpy())

        return loss

    def compute_ll(self, p, data):
        """ Compute log likelihood.

        Parameters
        ----------
        p: torch.distributions
            probabilistic decoder (or likelihood of generating true data
            sample given the latent code).
        data: torch.Tensor
            reference data.
        """
        return - p.log_prob(data).sum(-1, keepdim=True)

    def kl_normal_loss(self, q):
        """ Calculates the KL divergence between a normal distribution
        with diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        q: torch.distributions
            probabilistic encoder (or estimated posterior probability
            function).
        """
        dimension_wise_kl = kl_divergence(q, Normal(0, 1)).mean(dim=0)
        self.cache.setdefault("kl", []).append(
            dimension_wise_kl.detach().cpu().numpy())
        return dimension_wise_kl.sum()

    @staticmethod
    def kl_log_uniform(normal):
        """ Calculates the KL log uniform divergence.

        Paragraph 4.2 from:
        Variational Dropout Sparsifies Deep Neural Networks
        Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
        https://arxiv.org/abs/1701.05369
        https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/
        blob/master/KL%20approximation.ipynb
        """
        mu = normal.loc
        logvar = normal.scale.pow(2).log()
        log_alpha = BaseLoss.compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_kl = (k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 *
                  torch.log1p(torch.exp(-log_alpha)) - k1)
        return - neg_kl.mean(dim=0).sum()

    @staticmethod
    def compute_log_alpha(mu, logvar):
        return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(
            min=-8, max=8)

    def linear_annealing(self, init, fin):
        """ Linear annealing of a parameter.

        Returns
        -------
        annealed: float
            loss factor to gradually add the regularisation.
        """
        if self.steps_anneal == 0:
            return fin
        assert fin > init
        delta = fin - init
        annealed = min(init + delta * self.n_train_steps /
                       self.steps_anneal, fin)
        return annealed

    def update_train_step(self, iteration=None):
        """ Update the train step.

        Parameters
        ----------
        iteration: int, default None
            the current iteration.
        """
        if iteration is None and "iteration" in self.cache:
            iteration = self.cache["iteration"]
        if iteration is None:
            raise ValueError("No iteration specified.")
        self.n_train_steps = iteration


@Losses.register
class BetaHLoss(BaseLoss):
    """ Compute the Beta-VAE loss.

    beta-VAE: Learning Basic Visual Concepts with a Constrained Variational
    Framework, Irina Higgins, ICLR 2017.
    """
    def __init__(self, beta=4, **kwargs):
        """ Init class.

        Parameters
        ----------
        beta: float, default 4
            weight of the kl divergence.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super(BetaHLoss, self).__init__(**kwargs)
        self.beta = beta

    def __call__(self, p, data, **kwargs):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        kl_loss = self.kl_normal_loss(q)
        if model.training:
            anneal_reg = self.linear_annealing(init=0, fin=1)
            self.update_train_step()
        else:
            anneal_reg = 1
        kl_loss = anneal_reg * (self.beta * kl_loss)
        loss = rec_loss + kl_loss
        return loss, {"rec_loss": rec_loss, "kl_loss": kl_loss}


@Losses.register
class BetaBLoss(BaseLoss):
    """ Compute the Beta-VAE loss.

    Understanding disentangling in beta-VAE, Burgess, arXiv 2018.
    """
    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        """ Init class.

        Parameters
        ----------
        C_init: float, default 0
            starting annealed capacity C.
        C_fin: float, default 20
            final annealed capacity C.
        gamma: float, default 100
            weight of the KL divergence term.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super(BetaBLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin

    def __call__(self, p, data, **kwargs):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        kl_loss = self.kl_normal_loss(q)
        if model.training:
            C = self.linear_annealing(init=self.C_init, fin=self.C_fin)
            self.update_train_step()
        else:
            C = self.C_fin
        kl_loss = self.gamma * (kl_loss - C).abs()
        loss = rec_loss + kl_loss
        return loss, {"rec_loss": rec_loss, "kl_loss": kl_loss}


@Losses.register
class SparseLoss(BaseLoss):
    """ Compute the Beta-Sparse VAE loss.

    Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Luigi Antelmi, Nicholas Ayache, Philippe Robert,
    Marco Lorenzi, PMLR 2019.
    """
    def __init__(self, beta=4, **kwargs):
        """ Init class.

        Parameters
        ----------
        beta: float, default 4
            weight of the kl divergence.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super(SparseLoss, self).__init__(**kwargs)
        self.beta = beta

    def __call__(self, p, data, **kwargs):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        kl_loss = BaseLoss.kl_log_uniform(q)
        if model.training:
            anneal_reg = self.linear_annealing(init=0, fin=1)
        else:
            anneal_reg = 1
        kl_loss = anneal_reg * (self.beta * kl_loss)
        loss = rec_loss + kl_loss
        return loss, {"rec_loss": rec_loss, "kl_loss": kl_loss}


@Losses.register
class FactorKLoss(BaseLoss):
    """ Compute the Factor-VAE loss (algorithm 2).

    Disentangling by factorising, Hyunjik, arXiv 2018.
    """

    def __init__(self, device, gamma=10., disc_kwargs={},
                 optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)),
                 **kwargs):
        """ Init class.

        Parameters
        ----------
        device: torch.device
            the device.
        optimizer: torch.optim
            the network optimizer.
        gamma: float, default 10
            Weight of the TC loss term. `gamma` in the paper.
        disc_kwargs: dict
            discrimiator arguments.
        optim_kwargs: dict
            Adam optimizer arguments.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super(FactorKLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.device = device
        self.optimizer = optimier
        self.discriminator = Discriminator(**disc_kwargs).to(device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(),
                                      **optim_kwargs)

    def __call__(self, *args, **kwargs):
        """ Compute the loss.
        """
        raise NotImplementedError


@Losses.register
class BtcvaeLoss(BaseLoss):
    """ Compute the decomposed KL loss with either minibatch weighted
    sampling or minibatch stratified sampling according.

    Isolating sources of disentanglement in variational autoencoders, Tian Qi,
    Advances in Neural Information Processing Systems, 2018.
    """

    def __init__(self, dataset_size, alpha=1., beta=6., gamma=1.,
                 is_mss=True, **kwargs):
        """ Init class.

        Parameters
        ----------
        dataset_size: int
            number of training images in the dataset.
        alpha: float, default 1
            weight of the mutual information term.
        beta: float, default 6
            weight of the total correlation term.
        gamma: float, default 1
            weight of the dimension-wise KL term.
        dataset_size: int
            number of training images in the dataset.
        is_mss: bool, default True
            wether to use minibatch stratified sampling instead of minibatch
            weighted sampling.
        kwargs: dict
            additional arguments for 'BaseLoss'.
        """
        super(BtcvaeLoss, self).__init__(**kwargs)
        self.dataset_size = dataset_size
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.is_mss = is_mss

    def __call__(self, p, data):
        """ Compute the loss.
        """
        q, z, model = self.get_params()
        rec_loss = self.reconstruction_loss(p, data)
        log_pz, log_qz, log_prod_qzi, log_qz_x = self.get_probs(z, q)
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_qz_x - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()
        if model.training:
            anneal_reg = self.linear_annealing(init=0, fin=1)
            self.update_train_step()
        else:
            anneal_reg = 1
        mi_loss = self.alpha * mi_loss
        tc_loss = self.beta * tc_loss
        dw_kl_loss = anneal_reg * self.gamma * dw_kl_loss
        loss = rec_loss + mi_loss + tc_loss + dw_kl_loss
        return loss, {"mi_loss": mi_loss, "tc_loss": tc_loss,
                      "dw_kl_loss": dw_kl_loss}

    def get_probs(self, z, q):
        # Calculate log q(z|x)
        log_qz_x = q.log_prob(z).sum(dim=1)

        # Calculate log p(z)
        pz = Normal(loc=torch.zeros_like(z), scale=1)
        log_pz = pz.log_prob(z).sum(dim=1)

        # Calculate log q(z)
        batch_size = len(z)
        mat_log_qz = BtcvaeLoss.matrix_log_density_gaussian(z, q)
        if self.is_mss:
            log_iw_mat = BtcvaeLoss.log_importance_weight_matrix(
                batch_size, self.dataset_size)
            log_iw_mat = torch.unsqueeze(log_iw_mat, dim=-1).to(z.device)
            mat_log_qz = mat_log_qz + log_iw_mat
        log_qz = torch.logsumexp(mat_log_qz.sum(dim=2), dim=1, keepdim=False)
        log_prod_qzi = torch.logsumexp(
            mat_log_qz, dim=1, keepdim=False).sum(dim=1)

        return log_pz, log_qz, log_prod_qzi, log_qz_x

    @staticmethod
    def matrix_log_density_gaussian(x, q):
        """ Calculates log density of a Gaussian for all combination of bacth
        pairs of 'x' and 'mu', i.e. return tensor of shape (batch_size,
        batch_size, dim) instead of (batch_size, dim) in the usual log density.

        Parameters
        ----------
        x: torch.Tensor (batch_size, dim)
            value at which to compute the density.
        q: torch.distributions
            probabilistic encoder (or estimated posterior probability
            function).
        """
        x = torch.unsqueeze(x, dim=1)
        _mu = torch.unsqueeze(q.loc, dim=0)
        _sigma = torch.unsqueeze(q.scale, dim=0)
        _q = Normal(loc=_mu, scale=_sigma)
        return _q.log_prob(x)

    @staticmethod
    def log_importance_weight_matrix(batch_size, dataset_size):
        """ Calculates a log importance weight matrix.

        Parameters
        ----------
        batch_size: int
            number of training images in the batch.
        dataset_size: int
            number of training images in the dataset.
        """
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M + 1] = 1 / N
        W.view(-1)[1::M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()


@Losses.register
class VAEGMPLoss(object):
    """ VAEGMP Loss.
    """
    def __init__(self):
        """ Init class.
        """
        super(VAEGMPLoss, self).__init__()
        self.layer_outputs = None

    def __call__(self, p_x_given_z, data):
        """ Compute loss.
        """
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        q_z_given_x = self.layer_outputs["q_z_given_x"]
        z = self.layer_outputs["z"]
        p_z = self.layer_outputs["p_z"]

        # Reconstruction loss term i.e. the negative log-likelihood
        nll = - p_x_given_z.log_prob(data).sum()
        nll /= len(data)

        # Latent loss between approximate posterior and prior for z
        kl_div_z = (q_z_given_x.log_prob(z) - p_z.log_prob(z)).sum()
        kl_div_z /= len(data)

        # Need to maximise the ELBO with respect to these weights
        loss = nll + kl_div_z

        return loss, {"nll": nll, "kl_div_z": kl_div_z}


@Losses.register
class GMVAELoss(object):
    """ GMVAE Loss.
    """
    def __init__(self):
        """ Init class.
        """
        super(GMVAELoss, self).__init__()
        self.layer_outputs = None

    def __call__(self, p_x_given_z, data, labels=None):
        """ Compute loss.
        """
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        q_y_given_x = self.layer_outputs["q_y_given_x"]
        y = self.layer_outputs["y"]
        q_z_given_xy = self.layer_outputs["q_z_given_xy"]
        z = self.layer_outputs["z"]
        p_z_given_y = self.layer_outputs["p_z_given_y"]

        # Reconstruction loss term i.e. the negative log-likelihood
        nll = - p_x_given_z.log_prob(data).sum()
        nll /= len(data)

        # Latent loss between approximate posterior and prior for z
        kl_div_z = (q_z_given_xy.log_prob(z) - p_z_given_y.log_prob(z)).sum()
        kl_div_z /= len(data)

        # Conditional entropy loss
        logits = q_y_given_x.logits
        probs = func.softmax(logits, dim=-1)
        nent = (- probs * torch.log(probs)).sum()
        nent /= len(data)

        # Need to maximise the ELBO with respect to these weights
        loss = nll + kl_div_z + nent

        # Keep track of the clustering accuracy during training
        if labels is not None:
            cluster_acc = GMVAELoss.cluster_acc(
                q_y_given_x.logits, labels, is_logits=True)
        else:
            cluster_acc = 0

        return loss, {"nll": nll, "kl_div_z": kl_div_z, "nent": nent,
                      "cluster_acc": cluster_acc}

    @staticmethod
    def cluster_acc(y_pred, y, is_logits=False):
        # assert y_pred.size == y.size
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if is_logits:
            y_pred = np.argmax(y_pred, axis=1)
        n_classes = max(y_pred.max(), y.max()) + 1
        gain = np.zeros((n_classes, n_classes), dtype=np.int64)
        for idx in range(y_pred.size):
            gain[y_pred[idx], y[idx]] += 1
        cost = gain.max() - gain
        row_ind, col_ind = linear_sum_assignment(cost)
        return gain[row_ind, col_ind].sum() / y_pred.size


@Losses.register
class MCVAELoss(object):
    """ MCVAE loss.

    Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of
    Heterogeneous Data, Luigi Antelmi, Nicholas Ayache, Philippe Robert,
    Marco Lorenzi Proceedings of the 36th International Conference on Machine
    Learning, PMLR 97:302-311, 2019.

    MCVAE consists of two loss functions:

    1. KL divergence loss: how off the distribution over the latent space is
       from the prior. Given the prior is a standard Gaussian and the inferred
       distribution is a Gaussian with a diagonal covariance matrix,
       the KL-divergence becomes analytically solvable.
    2. log-likelihood LL

    loss = beta * KL_loss + LL_loss.
    """
    def __init__(self, n_channels, beta=1., enc_channels=None,
                 dec_channels=None, sparse=False, nodecoding=False):
        """ Init class.

        Parameters
        ----------
        n_channels: int
            the number of channels.
        beta, float, default 1.
            for beta-VAE.
        enc_channels: list of int, default None
            encode only these channels (for kl computation).
        dec_channels: list of int, default None
            decode only these channels (for ll computation).
        sparse: bool, default False
            use sparsity contraint.
        nodecoding: bool, default False
            if set do not apply the decoding.
        """
        super(MCVAELoss, self).__init__()
        self.n_channels = n_channels
        self.beta = beta
        self.sparse = sparse
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        if enc_channels is None:
            self.enc_channels = list(range(n_channels))
        else:
            assert(len(enc_channels) <= n_channels)
        if dec_channels is None:
            self.dec_channels = list(range(n_channels))
        else:
            assert(len(dec_channels) <= n_channels)
        self.n_enc_channels = len(self.enc_channels)
        self.n_dec_channels = len(self.dec_channels)
        self.nodecoding = nodecoding
        self.layer_outputs = None

    def __call__(self, p):
        """ Compute loss.

        Parameters
        ----------
        p: list of Normal distributions (C,) -> (N, F)
            reconstructed channels data.
        x: list of Tensor, (C,) -> (N, F)
            inputs channels data.
        """
        if self.nodecoding:
            return -1
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        x = self.layer_outputs["x"]
        q = self.layer_outputs["q"]
        kl = self.compute_kl(q, self.beta)
        ll = self.compute_ll(p, x)

        total = kl - ll
        return total, {"kl": kl, "ll": ll}

    def compute_kl(self, q, beta):
        kl = 0
        if not self.sparse:
            for c_idx, qi in enumerate(q):
                if c_idx in self.enc_channels:
                    kl += kl_divergence(qi, Normal(
                        0, 1)).sum(-1, keepdim=True).mean(0)
        else:
            for c_idx, qi in enumerate(q):
                if c_idx in self.enc_channels:
                    kl += self._kl_log_uniform(qi).sum(
                            -1, keepdim=True).mean(0)
        return beta * kl / self.n_enc_channels

    def compute_ll(self, p, x):
        # p[x][z]: p(x|z)
        ll = 0
        for c_idx1 in range(self.n_channels):
            for c_idx2 in range(self.n_channels):
                if c_idx1 in self.dec_channels and c_idx2 in self.enc_channels:
                    ll += self._compute_ll(
                        p=p[c_idx1][c_idx2], x=x[c_idx1]).mean(0)
        return ll / self.n_enc_channels / self.n_dec_channels

    def compute_log_alpha(self, mu, logvar):
        # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
        return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(
            min=-8, max=8)

    def _compute_ll(self, p, x):
        ll = p.log_prob(x).view(len(x), -1)
        return ll.sum(-1, keepdim=True)

    def _kl_log_uniform(self, normal):
        """
        Paragraph 4.2 from:
        Variational Dropout Sparsifies Deep Neural Networks
        Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
        https://arxiv.org/abs/1701.05369
        https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/
        blob/master/KL%20approximation.ipynb
        """
        mu = normal.loc
        logvar = normal.scale.pow(2).log()
        log_alpha = self.compute_log_alpha(mu, logvar)
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        neg_kl = (k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 *
                  torch.log1p(torch.exp(-log_alpha)) - k1)
        return - neg_kl


@Losses.register
class VaDELoss(object):
    """ VaDE loss.
    """
    def __init__(self, alpha=1):
        """ Init class.

        Parameters
        ----------
        alpha: float, default 1
            reconstruction loss weight.
        """
        super(VaDELoss, self).__init__()
        self.layer_outputs = None
        self.alpha = alpha

    def __call__(self, x_pred, x, *args, **kwargs):
        if self.layer_outputs is None:
            raise ValueError("The model needs to return the latent space "
                             "distribution parameters z, z_mu, z_var.")
        z_mu = self.layer_outputs["z_mu"]
        z_logvar = self.layer_outputs["z_logvar"]
        z = self.layer_outputs["z"]
        model = self.layer_outputs["model"]
        n_classes = model.n_classes
        batch_size = x.size(0)

        gamma = model.get_gamma(z, z_mu, z_logvar)
        z_mu_t = z_mu.unsqueeze(dim=2).expand(
            z_mu.size()[0], z_mu.size()[1], n_classes)
        z_logvar_t = z_logvar.unsqueeze(dim=2).expand(
            z_logvar.size()[0], z_logvar.size()[1], n_classes)
        u_t = model.u_p.unsqueeze(dim=0).expand(
            z.size()[0], model.u_p.size()[0], model.u_p.size()[1])
        lambda_t = model.lambda_p.unsqueeze(dim=0).expand(
            z.size()[0], model.lambda_p.size()[0], model.lambda_p.size()[1])
        theta_t = model.theta_p.unsqueeze(dim=0).expand(
            z.size()[0], n_classes)

        if model.binary:
            rec_loss = torch.sum(
                func.binary_cross_entropy(x_pred, x, reduction="none"), dim=1)
        else:
            rec_loss = torch.sum(
                func.mse_loss(x_pred, x, reduction="none"), dim=1)
        # rec_loss *= self.alpha * model.input_dim
        log_p_z_c = (
            torch.sum(
                0.5 * gamma
                * torch.sum((
                    math.log(2 * math.pi) + torch.log(lambda_t)
                    + torch.exp(z_logvar_t) / lambda_t
                    + (z_mu_t - u_t)**2 / lambda_t), dim=1), dim=1))
        q_entropy = (
            - 0.5 * torch.sum(1 + z_logvar + math.log(2 * math.pi), dim=1))
        log_p_c = - torch.sum(gamma * torch.log(theta_t), dim=1)
        log_q_c_x = torch.sum(gamma * torch.log(gamma), dim=1)

        # Normalize by same number of elements as in reconstruction
        loss = torch.mean(rec_loss + log_p_z_c + q_entropy + log_p_c +
                          log_q_c_x)

        return loss

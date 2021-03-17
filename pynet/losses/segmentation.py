# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides segmentation losses.
"""

# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, kl_divergence
from torch.autograd import Variable
from pynet.utils import Losses


# Global parameters
logger = logging.getLogger("pynet")


@Losses.register
class FocalLoss(object):
    """ Define a Focal Loss.

    Loss(pt) = −αt mt (1−pt)γ log(pt)

    where pt is the model's estimated probability for each class.

    When an example is misclassified and pt is small, the modulating factor
    is near 1 and the loss is unaffected. As pt goes to 1, the factor goes to
    0 and the loss for well-classified examples is down-weighted.
    The focusing parameter γ smoothly adjusts the rate at which easy examples
    are down-weighted. When γ= 0, the loss is equivalent to cross entropy, and
    as γ isincreased the effect of the modulating factor is likewise increased.
    For instance, with γ= 2, an example classified with pt= 0.9 would have
    100×lower loss compared with cross entropy and with pt≈0.968 it would have
    1000×lower loss.
    Then we use an α-balanced variant of the focal loss for addressing class
    imbalance with a weighting factor α ∈ [0,1]. In practice α may be set by
    inverse class frequency.

    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, n_classes, gamma=2, alpha=None, reduction="mean",
                 with_logit=True):
        """ Class instanciation.

        Parameters
        ----------
        n_classes: int
            the number of classes.
        gamma: float, default 2
            the focusing parameter >=0.
        alpha: float or list of float, default None
            if set use alpha-balanced variant of the focal loss.
        reduction: str, default 'mean'
            specifies the reduction to apply to the output: 'none' - no
            reduction will be applied, 'mean' - the sum of the output
            will be divided by the number of elements in the output, 'sum'
            - the output will be summed.
        with_logit: bool, default True
            apply the softmax logit function to the result.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.with_logit = with_logit
        self.eps = 1e-9
        alpha = alpha or 1
        if not isinstance(alpha, list):
            alpha = [alpha] * n_classes
        if len(alpha) != n_classes:
            raise ValueError("Invalid alphas size.")
        logger.debug("  alpha: {0}".format(alpha))
        self.alpha = torch.FloatTensor(alpha).view(-1, 1)
        # self.alpha = self.alpha / self.alpha.sum()
        self.debug("alpha", self.alpha)

    def __call__(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤target[i]≤C−1.
        """
        logger.debug("Focal loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        dim = output.dim()
        logger.debug("  n_batches: {0}".format(n_batch))
        logger.debug("  n_classes: {0}".format(n_classes))
        logger.debug("  dim: {0}".format(dim))
        if self.with_logit:
            output = func.softmax(output, dim=1)
        logit = output + self.eps
        self.debug("logit", logit)

        # Reshape data
        # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
        if dim > 2:
            logit = logit.view(n_batch, n_classes, -1)
            self.debug("logit", logit)
            logit = logit.permute(0, 2, 1).contiguous()
            self.debug("logit", logit)
            logit = logit.view(-1, n_classes)
            self.debug("logit", logit)
        target = torch.squeeze(target, dim=1)
        target = target.view(-1, 1)
        self.debug("target", target)

        # Create the labels one hot encoded tensor
        idx = target.data
        one_hot = torch.zeros(target.size(0), n_classes,
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(1, idx, 1.) + self.eps

        # Compute the focal loss
        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        pt = torch.sum(target_one_hot * logit, dim=1)
        self.debug("pt", pt)
        logpt = torch.log(pt)
        weight = torch.pow(1 - pt, self.gamma)
        self.debug("weight", weight)
        alpha = self.alpha[idx]
        alpha = torch.squeeze(alpha)
        self.debug("alpha", alpha)
        loss = -1 * alpha * weight * logpt
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss) / self.alpha[target].mean()
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def _forward_without_resizing(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤target[i]≤C−1.
        """
        logger.debug("Focal loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        dim = output.dim()
        logger.debug("  n_batches: {0}".format(n_batch))
        logger.debug("  n_classes: {0}".format(n_classes))
        logger.debug("  dim: {0}".format(dim))
        if self.with_logit:
            output = func.softmax(output, dim=1)
        logit = output + self.eps
        self.debug("logit", logit)

        # Create the labels one hot encoded tensor
        one_hot = torch.zeros(n_batch, n_classes, *target.shape[1:],
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(
            1, target.unsqueeze(1), 1.) + self.eps

        # Compute the focal loss
        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        weight = torch.pow(1 - logit, self.gamma)
        self.debug("weight", weight)
        shape = [1, n_classes] + [1] * len(target.shape[1:])
        alpha = self.alpha.view(*shape)
        alpha = alpha.expand_as(weight)
        self.debug("alpha", alpha)
        focal = -1 * alpha * weight * torch.log(logit)
        self.debug("focal", focal)
        loss = torch.sum(target_one_hot * focal, dim=1)
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss) / self.alpha[target].mean()
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class MaskLoss(object):
    """ Define a Masked Loss.

    Loss(pt) = −αt mt log(pt)

    where pt is the model's estimated probability for each class.
    """
    def __init__(self, n_classes, beta=0.2, alpha=None, reduction="mean",
                 with_logit=True):
        """ Class instanciation.

        Parameters
        ----------
        n_classes: int
            the number of classes.
        beta: float, default 0.2
            the minimum value in the mask.
        alpha: float or list of float, default None
            if set use alpha-balanced variant of the focal loss.
        reduction: str, default 'mean'
            specifies the reduction to apply to the output: 'none' - no
            reduction will be applied, 'mean' - the sum of the output
            will be divided by the number of elements in the output, 'sum'
            - the output will be summed.
        with_logit: bool, default True
            apply the log softmax logit function to the result.
        """
        self.beta = beta
        self.alpha = alpha
        self.reduction = reduction
        self.with_logit = with_logit
        self.eps = 1e-9
        alpha = alpha or 1
        if not isinstance(alpha, list):
            alpha = [alpha] * n_classes
        if len(alpha) != n_classes:
            raise ValueError("Invalid alphas size.")
        logger.debug("  alpha: {0}".format(alpha))
        self.alpha = torch.FloatTensor(alpha)
        self.debug("alpha", self.alpha)

    def __call__(self, output, target, mask):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤ target[i] ≤C−1.
        mask: Tensor (N,*)
            the binary mask used to mask the loss.
        """
        logger.debug("Maked loss...")
        self.debug("output", output)
        self.debug("target", target)
        self.debug("mask", mask)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")
        if mask is not None and output.shape[0] != mask.shape[0]:
            raise ValueError("Expected pred & mask same batch size.")
        if mask is not None and output.shape[2:] != mask.shape[1:]:
            raise ValueError("Expected pred & mask same data size.")
        if mask is not None and output.device != mask.device:
            raise ValueError("Pred & mask must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        logger.debug("  n_batches: {0}".format(n_batch))
        logger.debug("  n_classes: {0}".format(n_classes))

        if self.alpha.device != device:
            self.alpha = self.alpha.to(device)
        if self.with_logit:
            output = func.log_softmax(output, dim=1)
        logit = output + self.eps
        self.debug("logit", logit)

        # Compute the focal loss
        mask[mask <= self.beta] = self.beta
        loss = func.nll_loss(logit, target, weight=self.alpha,
                             reduction="none")
        loss = loss * mask
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss) / self.alpha[target].mean()
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class SoftDiceLoss(object):
    """ Define a multi class Dice Loss.

    Dice = (2 intersec Y) / (X + Y)

    Note that PyTorch optimizers minimize a loss. In this case, we would like
    to maximize the dice loss so we return 1 - Dice.
    """
    def __init__(self, with_logit=True, reduction="mean"):
        """ Class instanciation.

        Parameters
        ----------
        with_logit: bool, default True
            apply the softmax logit function to the result.
        reduction: str, default 'mean'
            specifies the reduction to apply to the output: 'none' - no
            reduction will be applied, 'mean' - the sum of the output
            will be divided by the number of elements in the output, 'sum'
            - the output will be summed.
        """
        self.with_logit = with_logit
        self.reduction = reduction
        self.smooth = 1e-6
        self.eps = 1e-6

    def __call__(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤targets[i]≤C−1.
        """
        logger.debug("Dice loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        if self.with_logit:
            prob = func.softmax(output, dim=1)
        else:
            prob = output
        self.debug("logit", prob)

        # Create the labels one hot encoded tensor
        prob = prob.view(n_batch, -1)
        dims = list(range(len(target.shape)))
        dims.insert(1, len(target.shape))
        dims = tuple(dims)
        logger.debug("permute {0}".format(dims))
        target_one_hot = func.one_hot(target, num_classes=n_classes)
        self.debug("target_one_hot", target_one_hot)
        target_one_hot = target_one_hot.permute(dims)
        target_one_hot = target_one_hot.contiguous().view(n_batch, -1)
        if target_one_hot.device != device:
            target_one_hot = target_one_hot.to(device)
        self.debug("target_one_hot", target_one_hot)

        # Compute the dice score
        intersection = prob * target_one_hot
        self.debug("intersection", intersection)
        dice_score = (2 * intersection.sum(dim=1) + self.smooth) / (
            target_one_hot.sum(dim=1) + prob.sum(dim=1) + self.smooth)
        loss = 1. - dice_score
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def _forward_without_resizing(self, output, target):
        """ Compute the loss.

        Parameters
        ----------
        output: Tensor (N,C,*)
            predicted labels where C is the number of classes.
        target: Tensor (N,*)
            true labels where each value is 0≤targets[i]≤C−1.
        """
        logger.debug("Dice loss...")
        self.debug("output", output)
        self.debug("target", target)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape[0] != target.shape[0]:
            raise ValueError("Expected pred & true labels same batch size.")
        if output.shape[2:] != target.shape[1:]:
            raise ValueError("Expected pred & true labels same data size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        n_batch, n_classes = output.shape[:2]
        device = output.device
        if self.with_logit:
            prob = func.softmax(output, dim=1)
        else:
            prob = output
        self.debug("logit", prob)

        # Create the labels one hot encoded tensor
        one_hot = torch.zeros(n_batch, n_classes, *target.shape[1:],
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(1, target.unsqueeze(1), 1.)
        self.debug("one hot", target_one_hot)

        # Compute the dice score
        dims = tuple(range(1, len(target.shape) + 1))
        intersection = torch.sum(prob * target_one_hot, dims)
        self.debug("intersection", intersection)
        cardinality = torch.sum(prob + target_one_hot, dims)
        self.debug("cardinality", cardinality)
        dice_score = 2. * intersection / (cardinality + self.eps)
        loss = 1. - dice_score
        self.debug("loss", loss)

        # Reduction
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        else:
            raise NotImplementedError("Invalid reduction mode.")
        logger.debug("  loss: {0}".format(loss))

        return loss

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))


@Losses.register
class CustomKLLoss(object):
    """ KL Loss.
    """
    def __init__(self, reduction="mean"):
        super(CustomKLLoss, self).__init__()
        self.reduction = reduction

    def __call__(self, posterior):
        kl_loss = kl_divergence(posterior, Normal(0, 1)).sum(-1, keepdim=True)
        if self.reduction == "mean":
            return kl_loss.mean(0)
        elif self.reduction == "sum":
            return kl_loss.sum(0)
        elif self.reduction == "none":
            return kl_loss
        else:
            return NotImplementedError


@Losses.register
class NvNetCombinedLoss(object):
    """ Combined Loss.

    Cross Entropy loss + k1 * L2 loss + k2 * KL loss
    Since the output of the segmentation decoder has N channels (prediction
    for each tumor subregion), we simply add the N dice loss functions.
    A hyper-parameter weight of k1=0.1, k2=0.1 was found empirically in the
    paper.
    """
    def __init__(self, num_classes, k1=0.1, k2=0.1):
        super(NvNetCombinedLoss, self).__init__()
        self.layer_outputs = None
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.l2_loss = nn.MSELoss(reduction="mean")
        self.kl_loss = CustomKLLoss(reduction="mean")

    def __call__(self, output, target):
        logger.debug("NvNet Combined Loss...")
        self.debug("output", output)
        self.debug("target", target)
        if self.layer_outputs is not None:
            z = self.layer_outputs["z"]
            posterior = self.layer_outputs["q"]
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape != target.shape:
            raise ValueError("Expected pred & true of same size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")

        device = output.device
        if self.layer_outputs is not None:
            vae_pred = output[:, self.num_classes:]
            vae_truth = target[:, self.num_classes:]
            self.debug("vae_pred", vae_pred)
            self.debug("vae_truth", vae_truth)
        seg_pred = output[:, :self.num_classes]
        seg_truth = target[:, :self.num_classes]
        self.debug("seg_pred", seg_pred)
        self.debug("seg_truth", seg_truth)
        seg_truth = torch.argmax(seg_truth, dim=1).type(torch.LongTensor)
        if seg_truth.device != device:
            seg_truth = seg_truth.to(device)
        self.debug("seg_truth", seg_truth)

        ce_loss = self.ce_loss(seg_pred, seg_truth)
        if self.layer_outputs is not None:
            l2_loss = self.l2_loss(vae_pred, vae_truth)
            kl_div = self.kl_loss(posterior)
            combined_loss = ce_loss + self.k1 * l2_loss + self.k2 * kl_div
        else:
            l2_loss, kl_div = 0, 0
            combined_loss = ce_loss
        logger.debug(
            "ce_loss: {0}, L2_loss: {1}, KL_div: {2}, combined_loss: "
            "{3}".format(ce_loss, l2_loss, kl_div, combined_loss))
        return combined_loss, {"l2_loss": l2_loss, "kl_loss": kl_div,
                               "ce_loss": ce_loss}

    def debug(self, name, tensor):
        """ Print debug message.

        Parameters
        ----------
        name: str
            the tensor name in the displayed message.
        tensor: Tensor
            a pytorch tensor.
        """
        logger.debug("  {3}: {0} - {1} - {2}".format(
            tensor.shape, tensor.get_device(), tensor.dtype, name))

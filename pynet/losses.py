# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
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
        print(prob)
        self.debug("logit", prob)

        # Create the labels one hot encoded tensor
        one_hot = torch.zeros(n_batch, n_classes, *target.shape[1:],
                              device=device, dtype=output.dtype)
        target_one_hot = one_hot.scatter_(1, target.unsqueeze(1), 1.)
        print(target_one_hot)
        self.debug("one hot", target_one_hot)

        # Compute the dice score
        dims = tuple(range(1, len(target.shape) + 1))
        intersection = torch.sum(prob * target_one_hot, dims)
        print(intersection)
        self.debug("intersection", intersection)
        cardinality = torch.sum(prob + target_one_hot, dims)
        print(cardinality)
        self.debug("cardinality", cardinality)
        dice_score = 2. * intersection / (cardinality + self.eps)
        print(dice_score)
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
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def __call__(self, mean, std):
        return (torch.mean(torch.mul(mean, mean)) +
                torch.mean(torch.mul(std, std)) -
                torch.mean(torch.log(torch.mul(std, std))) - 1)


@Losses.register
class NvNetCombinedLoss(object):
    """ Combined Loss.

    Diceloss + k1 * L2loss + k2 * KLloss
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
        self.ce_loss = nn.CrossEntropyLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def __call__(self, output, target):
        logger.debug("NvNet Combined Loss...")
        self.debug("output", output)
        self.debug("target", target)
        if self.layer_outputs is not None:
            y_mid = self.layer_outputs
            self.debug("y_mid", y_mid)
        if len(output.shape) < 2:
            raise ValueError("Invalid labels shape {0}.".format(output.shape))
        if output.shape != target.shape:
            raise ValueError("Expected pred & true of same size.")
        if output.device != target.device:
            raise ValueError("Pred & true labels must be in the same device.")
        if self.layer_outputs is not None and y_mid.shape[-1] != 256:
            raise ValueError("128 means & stds expected.")

        device = output.device
        if self.layer_outputs is not None:
            est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
            self.debug("est_mean", est_mean)
            self.debug("est_std", est_std)
            vae_pred = output[:, self.num_classes:]
            vae_truth = target[:, self.num_classes:]
            self.debug("vae_pred", vae_pred)
            self.debug("seg_truth", seg_truth)
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
            kl_div = self.kl_loss(est_mean, est_std)
            combined_loss = ce_loss + self.k1 * l2_loss + self.k2 * kl_div
        else:
            l2_loss, kl_div = (None, None)
            combined_loss = ce_loss
        logger.debug(
            "ce_loss: {0}, L2_loss: {1}, KL_div: {2}, combined_loss: "
            "{3}".format(ce_loss, l2_loss, kl_div, combined_loss))
        return combined_loss

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
class MSELoss(object):
    """ Calculate the Mean Square Error loss between I and J.
    """
    def __init__(self, concat=False):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        """
        super(MSELoss, self).__init__()
        self.concat = concat

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute MSE loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))
        loss = torch.mean((arr_i - arr_j) ** 2)
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss


@Losses.register
class PCCLoss(object):
    """ Calculate the Pearson correlation coefficient between I and J.
    """
    def __init__(self, concat=False):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        """
        super(PCCLoss, self).__init__()
        self.concat = concat

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute PCC loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        logger.debug("  channels: {0}".format(nb_channels))
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))
        centered_arr_i = arr_i - torch.mean(arr_i)
        centered_arr_j = arr_j - torch.mean(arr_j)
        pearson_loss = torch.sum(
            centered_arr_i * centered_arr_j) / (
                torch.sqrt(torch.sum(centered_arr_i ** 2) + 1e-6) *
                torch.sqrt(torch.sum(centered_arr_j ** 2) + 1e-6))
        loss = 1. - pearson_loss
        logger.debug("  loss: {0}".format(loss))
        logger.info("Done.")
        return loss


@Losses.register
class NCCLoss(object):
    """ Calculate the normalize cross correlation between I and J.
    """
    def __init__(self, concat=False, win=None):
        """ Init class.

        Parameters
        ----------
        concat: bool, default False
            if set asssume that the target image J is a concatenation of the
            moving and fixed.
        win: list of in, default None
            the window size to compute the correlation, default 9.
        """
        super(NCCLoss, self).__init__()
        self.concat = concat
        self.win = win

    def __call__(self, arr_i, arr_j):
        """ Forward method.

        Parameters
        ----------
        arr_i, arr_j: Tensor (batch_size, channels, *vol_shape)
            the input data.
        """
        logger.debug("Compute NCC loss...")
        if self.concat:
            nb_channels = arr_j.shape[1] // 2
            arr_j = arr_j[:, nb_channels:]
        ndims = len(list(arr_i.size())) - 2
        if ndims not in [1, 2, 3]:
            raise ValueError("Volumes should be 1 to 3 dimensions, not "
                             "{0}.".format(ndims))
        if self.win is None:
            self.win = [9] * ndims
        device = arr_i.get_device()
        sum_filt = torch.ones([1, 1, *self.win]).to(device)
        pad_no = math.floor(self.win[0] / 2)
        stride = tuple([1] * ndims)
        padding = tuple([pad_no] * ndims)
        logger.debug("  ndims: {0}".format(ndims))
        logger.debug("  stride: {0}".format(stride))
        logger.debug("  padding: {0}".format(padding))
        logger.debug("  filt: {0} - {1}".format(
            sum_filt.shape, sum_filt.get_device()))
        logger.debug("  win: {0}".format(self.win))
        logger.debug("  I: {0} - {1} - {2}".format(
            arr_i.shape, arr_i.get_device(), arr_i.dtype))
        logger.debug("  J: {0} - {1} - {2}".format(
            arr_j.shape, arr_j.get_device(), arr_j.dtype))

        var_arr_i, var_arr_j, cross = self._compute_local_sums(
            arr_i, arr_j, sum_filt, stride, padding)
        cc = cross * cross / (var_arr_i * var_arr_j + 1e-5)
        loss = -1 * torch.mean(cc)
        logger.debug("  loss: {0}".format(loss))
        logger.info("Done.")

        return loss

    def _compute_local_sums(self, arr_i, arr_j, filt, stride, padding):
        conv_fn = getattr(func, "conv{0}d".format(len(self.win)))
        logger.debug("  conv: {0}".format(conv_fn))

        arr_i2 = arr_i * arr_i
        arr_j2 = arr_j * arr_j
        arr_ij = arr_i * arr_j

        sum_arr_i = conv_fn(arr_i, filt, stride=stride, padding=padding)
        sum_arr_j = conv_fn(arr_j, filt, stride=stride, padding=padding)
        sum_arr_i2 = conv_fn(arr_i2, filt, stride=stride, padding=padding)
        sum_arr_j2 = conv_fn(arr_j2, filt, stride=stride, padding=padding)
        sum_arr_ij = conv_fn(arr_ij, filt, stride=stride, padding=padding)

        win_size = np.prod(self.win)
        logger.debug("  win size: {0}".format(win_size))
        u_arr_i = sum_arr_i / win_size
        u_arr_j = sum_arr_j / win_size

        cross = (sum_arr_ij - u_arr_j * sum_arr_i - u_arr_i * sum_arr_j +
                 u_arr_i * u_arr_j * win_size)
        var_arr_i = (sum_arr_i2 - 2 * u_arr_i * sum_arr_i + u_arr_i *
                     u_arr_i * win_size)
        var_arr_j = (sum_arr_j2 - 2 * u_arr_j * sum_arr_j + u_arr_j *
                     u_arr_j * win_size)

        return var_arr_i, var_arr_j, cross


@Losses.register
class RCNetLoss(object):
    """ RCNet Loss function.

    This loss needs intermediate layers outputs.
    Use a callback function to set the 'layer_outputs' class parameter before
    each evaluation of the loss function.
    If you use an interface this parameter is updated automatically?

    PCCLoss
    """
    def __init__(self):
        self.similarity_loss = PCCLoss(concat=True)
        self.layer_outputs = None

    def __call__(self, moving, fixed):
        logger.debug("Compute RCNet loss...")
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        stem_results = self.layer_outputs["stem_results"]
        for stem_result in stem_results:
            params = stem_result["stem_params"]
            if params["raw_weight"] > 0:
                stem_result["raw_loss"] = self.similarity_loss(
                    stem_result["warped"], fixed) * params["raw_weight"]
        loss = sum([
            stem_result["raw_loss"] * stem_result["stem_params"]["weight"]
            for stem_result in stem_results if "raw_loss" in stem_result])
        self.layer_outputs = None
        logger.debug("  loss: {0}".format(loss))
        logger.debug("Done.")
        return loss


@Losses.register
class VMILoss(object):
    """ Variational Mutual information loss function.

    Reference: http://bayesiandeeplearning.org/2018/papers/136.pdf -
               https://discuss.pytorch.org/t/help-with-histogram-and-loss-
               backward/44052/5
    """
    def get_positive_expectation(self, p_samples, average=True):
        log_2 = math.log(2.)
        Ep = log_2 - F.softplus(-p_samples)
        # Note JSD will be shifted
        if average:
            return Ep.mean()
        else:
            return Ep

    def get_negative_expectation(self, q_samples, average=True):
        log_2 = math.log(2.)
        Eq = F.softplus(-q_samples) + q_samples - log_2
        # Note JSD will be shifted
        if average:
            return Eq.mean()
        else:
            return Eq

    def __call__(self, lmap, gmap):
        """ The fenchel_dual_loss from the DIM code
        Reshape tensors dims to (N, Channels, chunks).

        Parameters
        ----------
        lmap: Tensor
            the moving data.
        gmap: Tensor
            the fixed data.
        """
        lmap = lmap.reshape(2, 128, -1)
        gmap = gmap.squeeze()

        N, units, n_locals = lmap.size()
        n_multis = gmap.size(2)

        # First we make the input tensors the right shape.
        l = lmap.view(N, units, n_locals)
        l = lmap.permute(0, 2, 1)
        l = lmap.reshape(-1, units)

        m = gmap.view(N, units, n_multis)
        m = gmap.permute(0, 2, 1)
        m = gmap.reshape(-1, units)

        u = torch.mm(m, l.t())
        u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

        mask = torch.eye(N).to(l.device)
        n_mask = 1 - mask

        E_pos = get_positive_expectation(u, average=False).mean(2).mean(2)
        E_neg = get_negative_expectation(u, average=False).mean(2).mean(2)

        E_pos = (E_pos * mask).sum() / mask.sum()
        E_neg = (E_neg * n_mask).sum() / n_mask.sum()
        loss = E_neg - E_pos

        return loss

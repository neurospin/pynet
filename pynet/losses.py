# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Third party import
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


def dice_loss_1(logits, true, eps=1e-7):
    """ Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    true: a tensor of shape [B, 1, H, W].
    logits: a tensor of shape [B, C, H, W]. Corresponds to
    the raw output or logits of the model.
    eps: added to the denominator for numerical stability.
    dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def dice_loss_2(output, target, weights=1):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    """
    output = func.softmax(output, dim=1)
    target = torch.argmax(target, dim=1).type(torch.LongTensor)
    encoded_target = output.data.clone().zero_()
    encoded_target.scatter_(1, target.unsqueeze(1), 1)
    encoded_target = Variable(encoded_target)

    assert output.size() == encoded_target.size(), "Input sizes must be equal."
    assert output.dim() == 4, "Input must be a 4D Tensor."

    num = (output * encoded_target).sum(dim=3).sum(dim=2)
    den1 = output.pow(2).sum(dim=3).sum(dim=2)
    den2 = encoded_target.pow(2).sum(dim=3).sum(dim=2)

    dice = (2 * num / (den1 + den2)) * weights
    return dice.sum() / dice.size(0)


class MultiDiceLoss(object):
    """ Define a multy classes dice loss.

    Note that PyTorch optimizers minimize a loss. In this case, we would like
    to maximize the dice loss so we return the negated dice loss.
    """
    def __init__(self, weight=None, ignore_index=None, nb_batch=None):
        """ Class instanciation.

        Parameters
        ----------
        weight: FloatTensor (C), default None
             a manual rescaling weight given to each class.
        ignore_index: int, default None
            specifies a target value that is ignored and does not contribute
            to the input gradient.
        nb_batch: int, default None
            the number of mini batch to rescale loss between 0 and 1.
        """
        self.weight = weight or 1
        self.ignore_index = ignore_index
        self.nb_batch = nb_batch or 1

    def __call__(self, output, target):
        """ Compute the loss.

        Note that this criterion is performing nn.Softmax() on the model
        outputs.

        Parameters
        ----------
        output: Variable (NxCxHxW)
            unnormalized scores for each class (the model output) where C is
            the number of classes.
        target: LongTensor (NxCxHxW)
            the class indices.
        """
        eps = 1  # 0.0001
        n_classes = output.size(1) * self.nb_batch

        output = func.softmax(output, dim=1)
        target = torch.argmax(target, dim=1).type(torch.LongTensor)
        # output = output.exp()

        encoded_target = output.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
        denominator = output + encoded_target
        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = self.weight * (1 - (numerator / denominator))
        print(loss_per_channel)

        return loss_per_channel.sum() / n_classes


class SoftDiceLoss(_Loss):
    """ Soft Dice Loss.
    """
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = (torch.sum(torch.mul(y_pred, y_pred)) +
                 torch.sum(torch.mul(y_true, y_true)) + eps)
        dice = 2 * intersection / union
        dice_loss = 1 - dice
        return dice_loss


class CustomKLLoss(_Loss):
    """ KL Loss.
    """
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return (torch.mean(torch.mul(mean, mean)) +
                torch.mean(torch.mul(std, std)) -
                torch.mean(torch.log(torch.mul(std, std))) - 1)


class CombinedLoss(_Loss):
    """ Combined Loss.

    Diceloss + k1 * L2loss + k2 * KLloss
    Since the output of the segmentation decoder has N channels (prediction
    for each tumor subregion), we simply add the N dice loss functions.
    A hyper-parameter weight of k1=0.1, k2=0.1 was found empirically in the
    paper.
    """
    def __init__(self, num_classes, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, outputs, y_true):
        y_pred, y_mid = outputs
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred = y_pred[:, :self.num_classes]
        seg_truth = y_true[:, :self.num_classes]
        vae_pred = y_pred[:, self.num_classes:]
        vae_truth = y_true[:, self.num_classes:]
        dice_loss = None
        for idx in range(self.num_classes):
            if dice_loss is None:
                dice_loss = self.dice_loss(
                    seg_pred[:, idx], seg_truth[:, idx])
            else:
                dice_loss += self.dice_loss(
                    seg_pred[:, idx], seg_truth[:, idx])
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        print("dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:"
              "%.4f" % (dice_loss, l2_loss, kl_div, combined_loss))
        return combined_loss

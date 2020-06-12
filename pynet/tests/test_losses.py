# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import numpy as np
import torch
import torch.nn as nn

# Package import
from pynet.losses import (
    FocalLoss, MaskLoss, SoftDiceLoss, MSELoss, PCCLoss, NCCLoss)


class TestLosses(unittest.TestCase):
    """ Test the losses defined in pynet.
    """
    def setUp(self):
        """ Setup test.
        """
        self.n_classes = 3
        self.x = torch.randn(2, self.n_classes, 3, 5, 5, requires_grad=True)
        self.target = torch.empty(2, 3, 5, 5, dtype=torch.long).random_(
            self.n_classes)
        self.weights = torch.tensor([1., 2., 3.])
        self.mask = torch.ones(2, 3, 5, 5)

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_focal(self):
        """ Test the FocalLoss.
        """
        criterion = FocalLoss(
            n_classes=self.n_classes, gamma=0, reduction="mean",
            with_logit=True, alpha=self.weights.numpy().tolist())
        ref_criterion = nn.CrossEntropyLoss(weight=self.weights)
        loss = criterion(self.x, self.target)
        ref_loss = ref_criterion(self.x, self.target)
        alt_loss = criterion._forward_without_resizing(self.x, self.target)
        loss.backward()
        self.assertTrue(np.allclose(
            loss.detach().numpy(), ref_loss.detach().numpy()))
        self.assertTrue(np.allclose(
            loss.detach().numpy(), alt_loss.detach().numpy()))

    def test_mask(self):
        """ Test the MaskLoss.
        """
        criterion = MaskLoss(
            n_classes=self.n_classes, beta=1., reduction="mean",
            with_logit=True, alpha=self.weights.numpy().tolist())
        ref_criterion = nn.CrossEntropyLoss(weight=self.weights)
        loss = criterion(self.x, self.target, self.mask)
        ref_loss = ref_criterion(self.x, self.target)
        loss.backward()
        self.assertTrue(np.allclose(
            loss.detach().numpy(), ref_loss.detach().numpy()))

    def test_softdice(self):
        """ Test the SoftDiceLoss.
        """
        criterion = SoftDiceLoss(reduction="mean", with_logit=True)
        loss = criterion(self.x, self.target)
        alt_loss = criterion._forward_without_resizing(self.x, self.target)
        loss.backward()
        self.assertTrue(np.allclose(
            loss.detach().numpy(), alt_loss.detach().numpy()))

    def test_mse(self):
        """ Test the MSELoss.
        """
        criterion = MSELoss()
        ref_criterion = nn.MSELoss(reduction="mean")
        loss = criterion(self.x, self.x)
        ref_loss = ref_criterion(self.x, self.x)
        loss.backward()
        self.assertTrue(np.allclose(
            loss.detach().numpy(), ref_loss.detach().numpy()))

    def test_pcc(self):
        """ Test the PCCLoss.
        """
        criterion = PCCLoss()
        loss = criterion(self.x, self.x)
        loss.backward()

    def test_ncc(self):
        """ Test the NCCLoss.
        """
        criterion = NCCLoss()
        loss = criterion(self.x[:, :1], self.x[:, :1])
        loss.backward()
        self.assertTrue(np.allclose(np.abs(loss.detach().numpy()), 1))


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()

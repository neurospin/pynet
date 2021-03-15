# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides losses.
"""

from .common import (
    MSELoss, PCCLoss, NCCLoss, RCNetLoss, VMILoss)
from .segmentation import (
    FocalLoss, MaskLoss, SoftDiceLoss, NvNetCombinedLoss)
from .generative import (
    MCVAELoss, BtcvaeLoss, SparseLoss, BetaBLoss, BetaHLoss)
from .generative import get_vae_loss

# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that privides common datasets.
"""


from pynet.utils import RegisteryDecorator


class Fetchers(RegisteryDecorator):
    """ Class that register all the available data fetchers.
    """
    REGISTRY = {}


from .core import DataManager
from .brats import fetch_brats
from .cifar import fetch_cifar
from .orientation import fetch_orientation
from .echocardiography import fetch_echocardiography
from .gradcam import fetch_gradcam
from .genomic import fetch_genomic_pred
from .registration import fetch_registration
from .ukb import fetch_height_biobank
from .impac import fetch_impac
from .connectome import fetch_connectome
from .hcp import fetch_hcp_brain
from .metastasis import fetch_metastasis
from .toy import fetch_toy


def get_fetchers():
    """ Return all available data fetchers.

    Returns
    -------
    fetchers: dict
        a dictionary containing all the fetchers.
    """
    return Fetchers.get_registry()

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

from .brats import fetch_brats
from .cifar import fetch_cifar
from .core import DataManager
from .orientation import fetch_orientation
from .echocardiography import fetch_echocardiography
from .gradcam import fetch_gradcam
from .genomic import fetch_genomic_pred
from .registration import fetch_registration
from .height_bb import fetch_height_biobank

# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Helper Module for Deep Learning
"""

import pynet.metrics
from .info import __version__
from .utils import setup_logging
from .utils import logo
from .utils import get_tools
from .interfaces import get_interfaces
from .interfaces import NetParameters

print(logo())
setup_logging(level="info")

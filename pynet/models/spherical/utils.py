# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides utility functions.
"""


# Imports
import logging


# Global parameters
logger = logging.getLogger("pynet")


def debug(name, tensor):
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

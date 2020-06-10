# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that privides common spatial and intensity normalization preprocessing
tools.
"""

# Imports
from collections import namedtuple
import logging
import nibabel
from .intensity import rescale
from .intensity import zscore_normalize
from .intensity import kde_normalize
from .spatial import reorient2std
from .spatial import biasfield
from .spatial import register
from .spatial import padd
from .spatial import downsample
from .spatial import scale


# Global parameters
logger = logging.getLogger("pynet")


class Processor(object):
    """ Class that can be used to register preprocessing steps.
    """
    Process = namedtuple("Process", ["process", "params", "apply_to"])

    def __init__(self):
        """ Initialize the class.
        """
        self.processes = []

    def register(self, process, apply_to, **kwargs):
        """ Register a new preprocessing step.

        Parameters
        ----------
        process: callable
            the preprocessing function.
        apply_to: str
            the registered preprocessing step expect 'image' or 'array' as
            input data.
        kwargs
            the preprocessing function parameters.
        """
        proc = self.Process(process=process, params=kwargs, apply_to=apply_to)
        self.processes.append(proc)

    def __call__(self, im):
        """ Apply the registered preprocessing steps.

        Parameters
        ----------
        im: nibabel.Nifti1Image
            the input image to be preprocessed.

        Returns
        -------
        normalized: nibabel.Nifti1Image
            the normalized input image.
        """
        for proc in self.processes:
            logger.debug("Applying {0}...".format(proc.process))
            if proc.apply_to == "image":
                im = proc.process(im, **proc.params)
            elif proc.apply_to == "array":
                arr = im.get_data()
                arr = proc.process(arr, **proc.params)
                im = nibabel.Nifti1Image(arr, im.affine)
            else:
                raise ValueError("Unsupported input data type.")
            logger.debug("Done.")
        return im

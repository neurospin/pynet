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

# Package import
from pynet.augmentation import add_blur
from pynet.augmentation import add_noise
from pynet.augmentation import add_ghosting
from pynet.augmentation import add_spike
from pynet.augmentation import add_biasfield
from pynet.augmentation import add_motion
from pynet.augmentation import add_offset
from pynet.augmentation import flip
from pynet.augmentation import affine
from pynet.augmentation import deformation
from pynet.augmentation import Transformer


class TestAugmentation(unittest.TestCase):
    """ Test the data augmentation defined in pynet.
    """
    def setUp(self):
        """ Setup test.
        """
        compose_transforms = Transformer(with_channel=False)
        compose_transforms.register(
            flip, probability=0.5, axis=0, apply_to=["all"])
        compose_transforms.register(
            add_blur, probability=1, sigma=4, apply_to=["all"])
        self.transforms = {
            "add_blur": (add_blur, {"sigma": 4}),
            "add_noise": (add_noise, {"snr": 5., "noise_type": "rician"}),
            "flip": (flip, {"axis": 0}),
            "affine": (affine, {
                "rotation": 5, "translation": 0, "zoom": 0.05}),
            "add_ghosting": (add_ghosting, {
                "n_ghosts": (4, 10), "axis": 2, "intensity": (0.5, 1)}),
            "add_spike": (add_spike, {"n_spikes": 1, "intensity": (0.1, 1)}),
            "add_biasfield": (add_biasfield, {"coefficients": 0.5}),
            "deformation": (deformation, {"max_displacement": 4, "alpha": 3}),
            "add_motion": (add_motion, {
                "rotation": 10, "translation": 10, "n_transforms": 2,
                "perturbation": 0.3}),
            "add_offset": (add_offset, {"factor": (0.05, 0.1)}),
            "compose_transforms": (compose_transforms, {}),
        }
        self.x = np.zeros((64, 64, 64))

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_transforms(self):
        """ Test the transforms.
        """
        for key, (fct, kwargs) in self.transforms.items():
            y = fct(self.x, **kwargs)


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()

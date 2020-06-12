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
import unittest.mock as mock
from unittest.mock import patch
import numpy as np
import nibabel

# Package import
from pynet.preprocessing import zscore_normalize
from pynet.preprocessing import kde_normalize
from pynet.preprocessing import reorient2std
from pynet.preprocessing import biasfield
from pynet.preprocessing import register
from pynet.preprocessing import downsample
from pynet.preprocessing import padd
from pynet.preprocessing import scale
from pynet.preprocessing import Processor


class TestPreprocessing(unittest.TestCase):
    """ Test the preprocessing steps defined in pynet.
    """
    def setUp(self):
        """ Setup test.
        """
        self.popen_patcher = patch(
            "pynet.preprocessing.spatial.subprocess.Popen")
        self.mock_popen = self.popen_patcher.start()
        mock_process = mock.Mock()
        attrs = {
            "communicate.return_value": (b"mock_OK", b"mock_NONE"),
            "returncode": 0
        }
        mock_process.configure_mock(**attrs)
        self.mock_popen.return_value = mock_process

        self.x = nibabel.Nifti1Image(np.random.rand(64, 64, 64), np.eye(4))
        pipeline = Processor()
        pipeline.register(reorient2std, apply_to="image")
        self.processes = {
            "zscore_normalize": (zscore_normalize, {"mask": None}),
            "kde_normalize": (kde_normalize, {
                "mask": None, "modality": "T1w", "norm_value": 1}),
            "reorient2std": (reorient2std, {}),
            "register": (register, {
                "target": self.x, "cost": "corratio",
                "interp": "trilinear", "dof": 6}),
            "biasfield": (biasfield, {"nb_iterations": 3}),
            "downsample": (downsample, {"scale": 2}),
            "padd": (padd, {"shape": [128, 128, 128], "fill_value": 0}),
            "pipeline": (pipeline, {})
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()

    @mock.patch("nibabel.load")
    @mock.patch("subprocess.check_call")
    def test_processes(self, mock_call, mock_load):
        """ Test the processes.
        """
        mock_load.return_value = self.x
        for key, (fct, kwargs) in self.processes.items():
            if key in ("reorient2std", "biasfield", "register", "pipeline"):
                y = fct(self.x, **kwargs).get_data()
            else:
                y = fct(self.x.get_data(), **kwargs)


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()

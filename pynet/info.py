# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


# Module current version
version_major = 0
version_minor = 0
version_micro = 0

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)


# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Helper Module for Deep Learning.
"""
SUMMARY = """
.. container:: summary-carousel

    pynet is a Python module that brings helper functions to:

    * design a new neural network from building blocks
    * generate a test, training and validation dataset.
    * train a model with metrics feedback.
    * investigate a trained deep network with common vizualization tools.
"""
long_description = """
=====
pynet
=====

Helper Module for Deep Learning.
"""

# Main setup parameters
NAME = "pynet"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/neurospin/pyney"
DOWNLOAD_URL = "https://github.com/neurospin/pynet"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = """
pynet developers
"""
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["pynet"]
REQUIRES = [
    "numpy>=1.17.1",
    "torch>=1.2.0",
    "torchvision>=0.4.0",
    "hiddenlayer>=0.2",
    "tabulate>=0.8.6",
    "matplotlib>=3.0.3",
    "nibabel>=2.4.0",
    "tqdm>=4.31.1",
    "progressbar2>=3.39.3",
    "requests>=2.9.1",
    "pandas>=0.24.2",
    "nilearn>=0.5.2",
    "Pillow>=6.2.1",
    "PySide2>=5.13.2",
    "scikit-learn>=0.21.3",
    "scikit-image>=0.15.0"
]
EXTRA_REQUIRES = {
}
SCRIPTS = [
]

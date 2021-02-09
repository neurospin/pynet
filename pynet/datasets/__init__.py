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


import logging
import numpy as np
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
from .tcga_lgg_tif import fetch_tcga_lgg_tif
from .primede import fetch_primede
from .minst import fetch_minst


# Global parameters
logger = logging.getLogger("pynet")


def get_fetchers():
    """ Return all available data fetchers.

    Returns
    -------
    fetchers: dict
        a dictionary containing all the fetchers.
    """
    return Fetchers.get_registry()


def get_data_manager(fetcher_name, datasetdir, static_fold=0, slicevol=False,
                     **kwargs):
    """ Return a ready to use data manager.

    Parameters
    ----------
    fetcher_name: str
        the name of the fetcher to be used.
    datasetdir: str
        the dataset destination folder.
    static_fold: int, default 0
        access the Kth fold < 10.
    slicevol: bool, default False
        in case of a 3d dataset, slice the volume to get a 2d dataset.
    kwargs: dict
        pass the DataManager parameters.

    Returns
    -------
    manager: DataManager
        the ready to use data manager.
    """
    fetchers = get_fetchers()
    fetcher = fetchers.get(fetcher_name)
    if fetcher is None:
        raise ValueError("Available fetchers are: {0}".format(
            list(fetchers.keys())))
    data = fetcher(datasetdir)
    if not hasattr(data, "metadata_path"):
        raise ValueError("One metadata path is expected.")
    if not hasattr(data, "input_path"):
        raise ValueError("One input path is expected.")
    if hasattr(data, "output_path"):
        output_path = data.output_path
    else:
        output_path = None
    if static_fold >= 10:
        raise ValueError("Fold index must be lower than 10")
    kwargs["number_of_folds"] = 10
    manager = DataManager(
        metadata_path=data.metadata_path, input_path=data.input_path,
        output_path=output_path, **kwargs)
    datasets = {
        "train": manager["train"][static_fold],
        "validation": manager["validation"][static_fold],
        "test": manager["test"]}
    if datasets["test"].outputs is None:
        raise ValueError("This code does not support label outputs yet.")
    inputs = {}
    for key, dataset in datasets.items():
        logger.debug("{0} indices: {1}".format(key, dataset.indices))
        data = [dataset.inputs[dataset.indices],
                dataset.outputs[dataset.indices]]
        logger.debug("X: {0}".format(data[0].shape))
        logger.debug("y: {0}".format(data[1].shape))
        if slicevol:
            for cnt, arr in enumerate(data):
                assert arr.ndim == 5, "Expect a 3D volume for slicing"
                shape = list(arr.shape[:-1])
                shape[0] = -1
                data[cnt] = arr.transpose(4, 0, 1, 2, 3).reshape(*shape)
                logger.debug("slice volume: {0}".format(data[cnt].shape))
        inputs["{0}_inputs".format(key)] = data[0]
        inputs["{0}_outputs".format(key)] = data[1]
    kwargs.update(inputs)
    for name in ("test_size", "sample_size", "number_of_folds", "sampler"):
        if name in kwargs:
            kwargs.pop(name)
    manager = DataManager.from_numpy(**kwargs)
    return manager

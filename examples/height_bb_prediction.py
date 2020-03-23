"""
Genome Wide Association with DL for standing height
===================================================

Credit: A Grigis

Load the data
-------------

Load some data.
You may need to change the 'datasetdir' parameter.
"""


import os
from pynet.datasets import DataManager, fetch_height_biobank
from pynet.utils import setup_logging

setup_logging(level="info")

data = fetch_height_biobank(datasetdir="/tmp/height_bb")
manager = DataManager(
    input_path=data.input_path,
    labels=["Height"],
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=5,
    test_size=0.2,
    continuous_labels=True)


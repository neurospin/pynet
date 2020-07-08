import os
from pynet.datasets import DataManager, fetch_aa_nicodep
from pynet.utils import setup_logging

setup_logging(level="info")

data = fetch_genomic_pred(
    datasetdir="/tmp/genomic_pred")
manager = DataManager(
    input_path=data.input_path,
    labels=["env0"],
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=5,
    test_size=0.2,
    continuous_labels=True)
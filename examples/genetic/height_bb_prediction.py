"""
Genome Wide Association with DL for standing height
===================================================

Credit: V Frouin

Load the data
-------------

Load some data.
You may need to change the 'datasetdir' parameter.
"""


import os
import sys
from pynet.datasets import DataManager, fetch_height_biobank
from pynet.utils import setup_logging

# This example cannot run in CI : it accesses NS intra filesystems
if "CI_MODE" in os.environ:
    sys.exit(0)

setup_logging(level="info")

data = fetch_height_biobank(datasetdir="/neurospin/tmp/height_bb")
manager = DataManager(
    input_path=data.input_path,
    labels=["Height"],
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=5,
    test_size=0.2,
    continuous_labels=True)


#############################################################################
# Basic inspection

import numpy as np
import matplotlib.pyplot as plt

train_dataset = manager["train"][0]
X_train = train_dataset.inputs[train_dataset.indices]
y_train = train_dataset.labels[train_dataset.indices]
test_dataset = manager["test"]
X_test = test_dataset.inputs[test_dataset.indices]
y_test = test_dataset.labels[test_dataset.indices]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("       min max mean sd")
print("Train:", y_train.min(), y_train.max(), y_train.mean(),
      np.sqrt(y_train.var()))
print("Test:", y_test.min(), y_test.max(), y_test.mean(),
      np.sqrt(y_test.var()))


#############################################################################
# SNP preselection according to a simple GWAS: select N_best most
# associated SNPs or select by min_P_value.
# Optional: not used after.

from scipy import stats

pvals = []
for idx in range(X_train.shape[1]):
    b, intercept, r_value, p_value, std_err = stats.linregress(
        X_train[:, idx], y_train)
    pvals.append(-np.log10(p_value))
pvals = np.array(pvals)
plt.figure()
plt.ylabel("-log10 P-value")
plt.xlabel("SNP")
plt.plot(pvals, marker="o")

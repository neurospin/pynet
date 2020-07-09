import os
from pynet.datasets import DataManager, fetch_aa_nicodep
from pynet.utils import setup_logging

setup_logging(level="info")

data = fetch_aa_nicodep()
manager = DataManager(
    input_path=data.input_path,
    labels=["smoker"],
    stratify_label="smoker",
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=5,
    test_size=0.2)

train_dataset = manager["train"][0]
X_train = train_dataset.inputs[train_dataset.indices]
y_train = train_dataset.labels[train_dataset.indices]
test_dataset = manager["test"]
X_test = test_dataset.inputs[test_dataset.indices]
y_test = test_dataset.labels[test_dataset.indices]
valid_dataset = manager["validation"][0]
X_valid = valid_dataset.inputs[valid_dataset.indices]
y_valid = valid_dataset.labels[valid_dataset.indices]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_valid.shape, y_valid.shape)
nb_snps = X_train.shape[1]
y_train = manager["train"][0].labels[train_dataset.indices]
print(y_train.shape)

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
import torch
import torch.nn as nn
import torch.nn.functional as func

# Package import
from pynet.interfaces import DeepLearningInterface
from pynet.datasets import DataManager, fetch_cifar


class TestCore(unittest.TestCase):
    """ Test the core of pynet.
    """
    def setUp(self):
        """ Setup test.
        """
        data = fetch_cifar(datasetdir="/tmp/cifar")
        self.manager = DataManager(
            input_path=data.input_path,
            labels=["label"],
            metadata_path=data.metadata_path,
            number_of_folds=10,
            batch_size=10,
            stratify_label="category",
            test_size=0.1,
            sample_size=0.01)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(func.relu(self.conv1(x)))
                x = self.pool(func.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = func.relu(self.fc1(x))
                x = func.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        self.cl = DeepLearningInterface(
            model=Net(),
            optimizer_name="SGD",
            momentum=0.9,
            learning_rate=0.001,
            loss_name="CrossEntropyLoss",
            metrics=["accuracy"])

    def tearDown(self):
        """ Run after each test.
        """
        pass

    def test_core(self):
        """ Test the core.
        """
        test_history, train_history = self.cl.training(
            manager=self.manager,
            nb_epochs=3,
            checkpointdir="/tmp/pynet",
            fold_index=0,
            with_validation=True)


if __name__ == "__main__":
    from pynet.utils import setup_logging
    setup_logging(level="debug")
    unittest.main()

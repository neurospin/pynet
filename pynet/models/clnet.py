# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Imports
import collections
import torch
import torch.nn as nn
import torch.nn.functional as func


# Classification
class Net(nn.Module):

    def __init__(self, nb_voxels_at_layer2, verbose=0):
        super(Net, self).__init__()
        self.nb_voxels_at_layer2 = nb_voxels_at_layer2
        self.conv1 = nn.Conv3d(1, 6, 5)
        #nn.init.xavier_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(6, 16, 5)
        self.linear1 = nn.Linear(16 * nb_voxels_at_layer2, 120)
        self.linear2 = nn.Linear(120, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = func.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * self.nb_voxels_at_layer2)
        x = func.relu(self.linear1(x))
        x = func.relu(self.linear2(x))
        return x


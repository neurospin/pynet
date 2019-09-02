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


class OneLayerMLP(nn.Module):
    """  Simple one hidden layer percetron.
    """
    def __init__(self, image_size, nb_neurons, nb_classes):
        """ Initialize the instance.

        Parameters
        ----------
        image_size: int
            the number of elemnts in the image.
        nb_neurons: int
            the number of neurons of the hidden layer.
        nb_classes: int
            the number of classes.
        """
        super(OneLayerMLP, self).__init__()
        self.layers = nn.Sequential(collections.OrderedDict([
            ("linear1", nn.Linear(image_size, nb_neurons)),
            ("relu1", nn.ReLU()),
            ("linear2", nn.Linear(nb_neurons, nb_classes)),
            ("softmax", nn.LogSoftmax(dim=1))
        ]))

    def forward(self, x): 
        x = self.layers(x)
        return x   

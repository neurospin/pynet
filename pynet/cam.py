# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that provides tools to compute class activation map.
"""


# Imports
import skimage
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func


class FeatureExtractor(object):
    """ Class for extracting activations and registering gradients from
    targetted intermediate layers.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs(object):
    """ Class for making a forward pass, and getting:
    1- the network output.
    2- activations from intermeddiate targetted layers.
    3- gradients from intermeddiate targetted layers.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(
            self.model.features, target_layers)

    def get_activations_gradient(self):
        return self.feature_extractor.gradients

    def get_activations(self, x):
        return self.feature_extractor(x)

    def __call__(self, x):
        if hasattr(self.model, "pre"):
            x = self.model.pre(x)
        target_activations, output = self.feature_extractor(x)
        if hasattr(self.model, "pool"):
            output = self.model.pool(output)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam(object):
    """ Class for computing class activation map.
    """
    def __init__(self, model, target_layers, labels, top=1):
        self.model = model
        self.labels = labels
        self.top = top
        self.model.eval()
        self.extractor = ModelOutputs(self.model, target_layers)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        features, output = self.extractor(input)
        pred_prob = func.softmax(output, dim=1).data.squeeze()
        probs, indices = pred_prob.sort(0, True)
        probs = probs.data.numpy()
        indices = indices.data.numpy()
        heatmaps = {}
        for cnt, (prob, index) in enumerate(zip(probs, indices)):
            if cnt == self.top:
                break
            label = self.labels[str(index)][1]
            line = "{0:.3f} -> {1}".format(prob, label)
            print(line)
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot * output)
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
            one_hot.backward(retain_graph=True)
            gradients = self.extractor.get_activations_gradient()[-1]
            gradients = gradients.cpu().data.numpy()
            pooled_gradients = np.mean(gradients, axis=(0, 2, 3))
            activations = features[-1]
            activations = activations.cpu().data.numpy()
            for cnt, weight in enumerate(pooled_gradients):
                activations[:, cnt] *= weight
            heatmap = np.mean(activations, axis=1).squeeze()
            heatmap = np.maximum(heatmap, 0)
            heatmap -= np.min(heatmap)
            heatmap /= np.max(heatmap)
            heatmap_highres = skimage.transform.resize(
                heatmap, input.shape[2:])
            heatmaps[label] = (input, heatmap, heatmap_highres)
        return heatmaps

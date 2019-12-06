"""
pynet: class activation map
===========================

Credit: A Grigis

Based on:

- https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
- http://snappishproductions.com/blog/2018/01/03/class-activation-mapping-in-pytorch.html
- https://github.com/jacobgil/pytorch-grad-cam

A class activation map for a particular category indicates the discriminative
image regions used by the CNN to identify that category.It provides us with
a way to look into what particular parts of the image influenced the whole
model's decision for a specifically assigned label.
It is particularly useful in analyzing wrongly classified samples.

It starts with finding the gradient of the most dominant logit with respect
to the latest activation map in the model. We can interpret this as some
encoded features that ended up activated in the final activation map
persuaded the model as a whole to choose that particular logit
(subsequently the corresponding class). The gradients are then pooled
channel-wise, and the activation channels are weighted with the corresponding
gradients, yielding the collection of weighted activation channels. By
inspecting these channels, we can tell which ones played the most significant
role in the decision of the class.

The main idea is to dissect the network as follows:

- load the model
- find its last convolutional layer
- compute the most probable class
- take the gradient of the class logit with respect to the activation maps we
  have just obtained
- pool the gradients
- weight the channels of the map by the corresponding pooled gradients
- interpolate the heat-map

We can compute the gradients in PyTorch, using the 'backward' method called on
a torch.Tensor. This is exactly what we are going to do: call 'backward()' on
the most probable logit, which we obtain by performing the forward pass of
the image through the network. However, PyTorch only caches the gradients of
the leaf nodes in the computational graph, such as weights, biases and other
parameters. The gradients of the output with respect to the activations are
merely intermediate values and are discarded as soon as the gradient propagates
through them on the way back. We will have to register the backward hook to
the activation map of the last convolutional layer in our model.

Load the data
-------------

Load some images and apply the ImageNet transformation.
You may need to change the 'datasetdir' parameter.
"""

from pynet.datasets import DataManager, fetch_gradcam
from pynet.plotting import plot_data

data = fetch_gradcam(
    datasetdir="/neurospin/nsap/datasets/gradcam")
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=5,
    test_size=1)
dataset = manager["test"]
print(dataset.inputs.shape)
plot_data(dataset.inputs, nb_samples=5, random=False, rgb=True)


#############################################################################
# Explore different architectures
# -------------------------------
#
# Let's automate this procedure for different networks.
# We need to reload the data for the inception network.
# You may need to change the 'datasetdir' parameter.

from pynet.models.cam import get_cam_network
from pynet.cam import GradCam
import matplotlib.pyplot as plt

data = fetch_gradcam(
    datasetdir="/neurospin/nsap/datasets/gradcam")
manager1 = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=1,
    test_size=1)
loaders1 = manager1.get_dataloader(test=True)
data = fetch_gradcam(
    datasetdir="/neurospin/nsap/datasets/gradcam",
    inception=True)
manager2 = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    number_of_folds=2,
    batch_size=1,
    test_size=1)
loaders2 = manager2.get_dataloader(test=True)

for loaders, model_name in ((loaders1, "vgg19"), (loaders1, "densenet201"),
                            (loaders1, "resnet18"), (loaders2, "inception_v3")):

    heatmaps = []
    print("-" * 10)
    print(model_name)
    for dataitem in loaders.test:
        model, activation_layer_name = get_cam_network(model_name)
        grad_cam = GradCam(model, [activation_layer_name], data.labels, top=1)
        heatmaps.extend(grad_cam(dataitem.inputs).items())

    fig, axs = plt.subplots(nrows=2, ncols=len(heatmaps))
    fig.suptitle(model_name, fontsize="large")
    for cnt, (name, (img, arr, arr_highres)) in enumerate(heatmaps):
        axs[0, cnt].set_title(name)
        axs[0, cnt].matshow(arr)
        axs[0, cnt].set_axis_off()
        _img = img.data.numpy()[0].transpose((1, 2, 0))
        axs[1, cnt].imshow(_img)
        axs[1, cnt].imshow(arr_highres, alpha=0.6, cmap="jet")
        axs[1, cnt].set_axis_off()

# plt.show()

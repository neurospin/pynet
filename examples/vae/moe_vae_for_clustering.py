"""
MoE-Sim-VAE
===========

Credit: A Grigis

Mixture of Experts VAE with similarity prior: MoE-Sim-VAE

Reference: Mixture-of-Experts Variational Autoencoder for Clustering and
Generating from Similarity-Based Representations on Single Cell Data,
Andreas Kopf, arXiv 2020.
"""

# Imports
import os
import sys
import sys
if "CI_MODE" in os.environ:
    sys.exit()
import numpy as np
import matplotlib.colors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import umap
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Normal, kl_divergence
import pynet
from pynet import NetParameters
from pynet.datasets import DataManager, fetch_minst
from pynet.interfaces import MOESimVAENetEncoder
from pynet.plotting import Board, update_board


#############################################################################
# Parameters
# ----------
#
# Define some global parameters that will be used to create and train the
# model:

random_state = 42
datasetdir = "/neurospin/nsap/datasets/minst"
checkpointdir = None
input_dim = 28 * 28
n_components_umap = 2
n_neighbors_knn = 10
batch_size = 128
n_epochs = 10 #20000
learning_rate = 0.0001
dropout_rate = 0.5
latent_dim = 68
n_experts = 10
beta = 1.
alpha = 1.
device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = pynet.get_tools(tool_name="losses")


#############################################################################
# MNIST dataset
# -------------
#
# The model will be trained on MNIST - handwritten digits dataset. The input
# is an image in R(28×28):

def flatten(arr):
    return arr.flatten()

data = fetch_minst(datasetdir=datasetdir)
manager = DataManager(
    input_path=data.input_path,
    metadata_path=data.metadata_path,
    stratify_label="label",
    labels="label",
    number_of_folds=10,
    batch_size=batch_size,
    test_size=0,
    input_transforms=[flatten],
    add_input=True,
    sample_size=1)


#############################################################################
# Data driven similarity matrix
# -----------------------------
#
# The similarity matrix is derived in an unsupervised way (eg, UMAP
# projection of the data and k nearest neighbors or distance thresholding to
# define the adjacency matrix for the batch), but can also be used to include
# weakly supervised information (eg, knowledge about diseased vs
# non diseased patients). The similarity feature in MoE Sim VAE can also be
# used to include prior knowledge about the best similarity measure on the
# data.

data = manager.inputs[:batch_size]
labels = manager.labels[:batch_size]
similarity, embedding = losses["MOESimVAELoss"].get_similarity_matrix(
    data, n_components_umap, n_neighbors_knn, random_state=random_state)
print("-- umap embedding:", embedding.shape)
print("-- similarity:", similarity.shape)
fig, ax_array = plt.subplots(10, 10)
axes = ax_array.flatten()
for idx, ax in enumerate(axes):
    ax.imshow(data[idx, 0], cmap="gray_r")
plt.setp(axes, xticks=[], yticks=[], frame_on=False)
plt.tight_layout(h_pad=0.5, w_pad=0.01)
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="Spectral", s=5)
plt.gca().set_aspect("equal", "datalim")
plt.colorbar(boundaries=(np.arange(11) - 0.5)).set_ticks(np.arange(10))
plt.axis("off")
plt.title("UMAP projection of the dataset", fontsize=10)
plt.figure()
cmap = matplotlib.colors.ListedColormap(["white", "orange"])
plt.imshow(similarity, cmap=cmap)
plt.axis("off")
plt.title("K-nearest-neighbors graph", fontsize=10)
plt.colorbar(boundaries=(np.arange(3) - 0.5)).set_ticks(np.arange(2))


#############################################################################
# Similarity loss
# ---------------
#
# Reconstruct a data-driven clustering loss. We use the use case proposed in
# 'Understanding binary cross-entropy / log loss: a visual explanation'
# https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-
# a-visual-explanation-a3ac6025181a

x = np.array([-2.2, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6])
y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

custom_lines = [Line2D([0], [0], color="red", lw=4),
                Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="blue", lw=4)]

logr = LogisticRegression(solver="lbfgs")
logr.fit(x.reshape(-1, 1), y)
y_pred = logr.predict_proba(x.reshape(-1, 1))[:, 1].ravel()
prob = y_pred.copy()
prob[y == 0.] = 1 - prob[y == 0.]
loss = log_loss(y, y_pred)
print("x = {}".format(x))
print("y = {}".format(y))
print("p(y) = {}".format(np.round(y_pred, 2)))
print("Log Loss / Cross Entropy = {:.4f}".format(loss))

fig, ax = plt.subplots()
colors = ["red" if yi == 0. else "green" for yi in y]
ax.bar(x, -np.log(prob), width=0.1, color=colors, alpha=0.5)
ax.axhline(y=loss, color="black", linestyle="--")
ax.scatter(x, [-0.05 if yi == 0. else -1.15 for yi in y], color=colors,
           edgecolors="black", s=40, marker="o", alpha=0.5)
ax.plot(x, y_pred - 1.1, color="blue")
ax.bar(x[y == 1.], y_pred[y == 1.], width=0.1, bottom=-1.1, color="green",
       alpha=0.5)
ax.bar(x[y == 0.], 1 - y_pred[y == 0.], width=0.1,
       bottom=-(1. - y_pred[y == 0.] + 0.1), color="red", alpha=0.5)
ax.text(0.5, 0.5, "{:.4f}".format(loss))
ax.set_title("Binary Cross Entropy", fontsize=10)
ax.text(-0.45, 0.75, "-log(p)")
ax.text(-0.2, -0.3, "p")
ax.spines["left"].set_position("zero")
ax.spines["right"].set_color("none")
ax.yaxis.tick_left()
ax.spines["bottom"].set_position("zero")
ax.spines["top"].set_color("none")
ax.xaxis.tick_bottom()
ax.grid(True, which="both")
ax.legend(custom_lines, ["Negative", "Positive", "Sigmoid"])

probs_true = torch.zeros((4, 2))
probs_true[:2, 0] = 1
probs_true[2:, 1] = 1
similarity = torch.mm(probs_true, torch.transpose(probs_true, 0, 1))
factors = np.linspace(0.1, 1, 10)
sim_losses = []
ce_losses = []
print(similarity)
def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce
for factor in factors:
    probs = probs_true * factor
    probs[:2, 1] = 1 - factor
    probs[2:, 0] = 1 - factor
    predictions = torch.mm(probs, torch.transpose(probs, 0, 1))
    print(predictions)
    print(cross_entropy(predictions.numpy(), similarity.numpy()))
    _loss = losses["MOESimVAELoss"].similarity(probs, similarity)
    _loss = torch.mean(torch.sum(_loss, dim=1), dim=0)
    _ce_loss = log_loss(probs_true[:, 0], probs[:, 0])
    if np.isnan(_ce_loss):
        _ce_loss = 0.
    print(probs)
    print(_loss, _ce_loss)
    sim_losses.append(_loss.cpu().numpy())
    ce_losses.append(_ce_loss)
fig, ax = plt.subplots()
ax.plot(factors, sim_losses, color="blue", label="SIM")
ax.plot(factors, ce_losses, color="green", label="CE")
ax.set_title("SIMILARITY losses", fontsize=10)
ax.set_xlabel("factors")
ax.grid(True, which="both")
ax.legend()


#############################################################################
# DEPICT loss
# -----------
#
# The DEPICT loss encourages the model to learn invariant features
# from the latent representation for clustering with respect to noise.

probs = torch.ones((1, 10))
factors = np.linspace(0.1, 1, 10)
depict_losses = []
for factor in factors:
    probs_noisy = torch.ones((1, 10)) * factor
    _loss = losses["MOESimVAELoss"].depict(probs, probs_noisy).mean()
    depict_losses.append(_loss.cpu().numpy())
fig, ax = plt.subplots()
ax.plot(factors, depict_losses)
ax.set_title("DEPICT losses", fontsize=10)
ax.set_xlabel("factors")


#############################################################################
# The Model
# ---------
#
# The model is a VAE with a Gaussian Mixture Prior (GMP) and N independent
# decoder path:

params = NetParameters(
    input_dim=input_dim,
    latent_dim=latent_dim,
    n_mix_components=n_experts,
    dense_hidden_dims=[256],
    classifier_hidden_dims=[100],
    sigma_min=0.001,
    raw_sigma_bias=0.25,
    gen_bias_init=0,
    dropout=0.5)
interface = MOESimVAENetEncoder(
    params,
    optimizer_name="Adam",
    learning_rate=learning_rate,
    loss=losses["MOESimVAELoss"](
        beta=beta, alpha=alpha, n_components_umap=n_components_umap,
        n_neighbors_knn=n_neighbors_knn),
    use_cuda=(device.type != "cpu"))
print(interface.model)
interface.board = Board(
    port=8097, host="http://localhost", env="moevae")
interface.add_observer("after_epoch", update_board)
train_history, valid_history = interface.training(
    manager=manager,
    nb_epochs=n_epochs,
    checkpointdir=checkpointdir,
    save_after_epochs=100,
    fold_index=0,
    with_validation=False)
print(train_history)

plt.show()

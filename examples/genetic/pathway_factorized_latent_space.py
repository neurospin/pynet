"""
Pathway factorized latent space
===============================

Credit: A Grigis
"""

# Imports
import os
import sys
if "CI_MODE" in os.environ:
    sys.exit()
import shutil
import subprocess
from itertools import product
import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
import torch
import pynet
from pynet import NetParameters
from pynet.datasets import DataManager, fetch_kang
from pynet.interfaces import PMVAEEncoder
from pynet.utils import setup_logging



#############################################################################
# Parameters
# -----------------
#
# Define some global parameters that will be used to create and train the
# model:

datasetdir = "/neurospin/nsap/datasets/kang"
batch_size = 256
latent_dim = 4
nb_epochs = 1201
learning_rate = 0.001
beta = 1e-5
checkpointdir = "/neurospin/nsap/datasets/kang/checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = pynet.get_tools(tool_name="losses")
setup_logging(level="info")


#############################################################################
# Kang dataset
# ------------
#
# Fetch & load the Kang dataset:

data, trainset, testset, membership_mask = fetch_kang(
    datasetdir=datasetdir, random_state=0)
gtpath = os.path.join(datasetdir, "kang_recons.h5ad")
manager = DataManager.from_numpy(
    train_inputs=trainset, validation_inputs=testset, test_inputs=data,
    batch_size=batch_size, sampler="random", add_input=True)


#############################################################################
# Training
# --------
#
# Create/train the model:

if checkpointdir is not None:
    weights_filename = os.path.join(
        checkpointdir, "model_0_epoch_{0}.pth".format(nb_epochs - 1))
params = NetParameters(
    membership_mask=membership_mask,
    latent_dim=latent_dim,
    hidden_layers=[12],
    add_auxiliary_module=True,
    terms=membership_mask.index,
    activation=None)
if checkpointdir is not None and os.path.isfile(weights_filename):
    model = PMVAEEncoder(
        params,
        optimizer_name="Adam",
        learning_rate=learning_rate,
        loss=losses["PMVAELoss"](beta=beta),
        use_cuda=(device.type != "cpu"),
        pretrained=weights_filename)
    print(model.model)
else:
    model = PMVAEEncoder(
        params,
        optimizer_name="Adam",
        learning_rate=learning_rate,
        loss=losses["PMVAELoss"](beta=beta),
        use_cuda=(device.type != "cpu"))
    print(model.model)
    train_history, valid_history = model.training(
        manager=manager,
        nb_epochs=nb_epochs,
        checkpointdir=checkpointdir,
        save_after_epochs=100,
        fold_index=0,
        with_validation=True)


#############################################################################
# Reduce the number of dimensions
# -------------------------------
#
# Use TSNE to create a 2d representation of the results:

def extract_pathway_cols(df, pathway):
    mask = df.columns.str.startswith(pathway + "-")
    return df.loc[:, mask]


def compute_tsnes(recons, pathways):
    for key in pathways:
        tsne = TSNE(n_components=2)
        codes = extract_pathway_cols(recons.obsm["codes"], key)
        tsne = pd.DataFrame(
            TSNE().fit_transform(codes.values),
            index=recons.obs_names,
            columns=["{0}-0".format(key), "{0}-1".format(key)])
        yield tsne


output_file = os.path.join(checkpointdir, "kang_recons.h5ad")
generated_pathways = [
    "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM",
    "REACTOME_TCR_SIGNALING",
    "REACTOME_CELL_CYCLE"]
if not os.path.isfile(output_file):
    y, X, _, loss, values = model.testing(
        manager=manager,
        with_logit=False,
        predict=False,
        concat_layer_outputs="z")
    print(y.shape)
    global_recon = y[:, :membership_mask.shape[1]]
    z = y[:, membership_mask.shape[1]:]
    print(" -- global recon:", global_recon.shape)
    print(" -- z:", z.shape)
    recons = anndata.AnnData(
        pd.DataFrame(
            global_recon,
            index=data.obs_names,
            columns=data.var_names),
        obs=data.obs,
        varm=data.varm,
    )
    recons.obsm["codes"] = pd.DataFrame(
        z,
        index=data.obs_names,
        columns=model.model.latent_space_names())
    recons.obsm["pathway_tsnes"] = pd.concat(
        compute_tsnes(recons, generated_pathways),
        axis=1)
    recons.write(output_file)


#############################################################################
# Display
# --------
#
# Display the results & the ground truth:

def extract_pathway_cols(df, pathway):
    mask = df.columns.str.startswith(pathway + "-")
    return df.loc[:, mask]


def tab20(arg):
    cmap = plt.get_cmap("tab20")
    return rgb2hex(cmap(arg))


generated_recons = anndata.read(output_file)
recons = anndata.read(gtpath)
cmap = {
    "CD4 T": tab20(0),
    "CD8 T": tab20(1),
    "CD14 Mono": tab20(2),
    "CD16 Mono": tab20(3),
    "B": tab20(4),
    "DC": tab20(6),
    "NK": tab20(8),
    "T": tab20(10)}
pathways = [
    "INTERFERON_ALPHA_BETA_SIGNALIN",
    "CYTOKINE_SIGNALING_IN_IMMUNE_S",
    "TCR_SIGNALING",
    "CELL_CYCLE"]
for _name, _recons, _pathways in (
        ("GT", recons, pathways),
        ("GENERATED", generated_recons, generated_pathways)):
    fig, axes = plt.subplots(2, len(pathways), figsize=(6 * len(_pathways), 8))
    fig.suptitle("{0} pathway factorized latent space results".format(_name),
                 fontsize=15, y=0.99)
    pairs = product(["stimulated", "control"], _pathways)
    for ax, (active, key) in zip(axes.ravel(), pairs):
        mask = (_recons.obs["condition"] == active)
        codes = extract_pathway_cols(_recons.obsm["pathway_tsnes"], key)
        # plot non-active condition
        ax.scatter(*codes.loc[~mask].T.values, s=1, c="lightgrey", alpha=0.1) 
        # plot active condition
        ax.scatter(*codes.loc[mask].T.values,
                   c=list(map(cmap.get, _recons.obs.loc[mask, "cell_type"])),
                   s=1, alpha=0.5,)
        key = key.replace("REACTOME_", "")[:30]
        ax.set_title("{0} {1}".format(key, active), fontsize=10)
        ax.axis("off")
    fig.legend(
        handles=[Patch(color=c, label=l) for l,c in cmap.items()],
        ncol=4, loc=("lower center"), bbox_to_anchor=(0.5, 0.01),
        fontsize="xx-large", prop={"size": 10})
    plt.tight_layout()
    fig.subplots_adjust(bottom=.1)

plt.show()

